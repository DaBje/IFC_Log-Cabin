"""IFC Log Cabin - generate scribe-fit, saddle-notched log cabins in Blender
and hand them to Bonsai as IFC elements.

Run this file from Blender's Scripting workspace ("Run Script") to register it
for the current session.

How the geometry is built
-------------------------
Every cut is *predefined* from the log schedule. No log is intersected against
its neighbours, and there are no boolean operations: a log knows what it must
look like purely from its diameters, its length, and which course it sits in.

Courses alternate direction - even indices run along X, odd along Y - and sit
half a round apart, so perpendicular walls interlock. Two logs in the same wall
are therefore `round_rise = (r_butt + r_top) - scribe_depth` apart.

The key identity is the one real log builders rely on. Butt and top ends
alternate, so the log directly below runs the opposite taper, and

    r_log(s) + r_below(s) = r_butt + r_top    for every s

is constant along the length. The lateral groove depth is
`r_log + r_below - round_rise = scribe_depth` everywhere, with no dependence on
position. That is why alternating ends makes a wall rise level, and it is what
lets the groove be cut from a formula instead of from the neighbour's surface.

Each cross-section is a circle whose underside is replaced by the upper
envelope of:

  * the lateral groove - an arc of the log below, offset down by round_rise;
  * a saddle notch - a *horizontal line*, because the distance to a
    perpendicular axis is sqrt(along^2 + dz^2) and so does not depend on the
    third coordinate. A slab removed from a disc, exact at any depth.

Both curves are functions of the horizontal coordinate, so the outline is built
with fixed topology - a fixed count of points over the top, a fixed count along
the floor - and never needs resampling.
"""

bl_info = {
    "name": "IFC Log Cabin",
    "author": "David Bjelland",
    "version": (0, 4, 1),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Log Cabin",
    "description": "Generate scribe-fit log cabins and export them as IFC via Bonsai",
    "category": "Add Mesh",
}

import math
import random

import bmesh
import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
)
from mathutils import Vector

# How far a knot fades out before it reaches worked timber: a span of sweep in
# radians, and a share of the radius below a sawn flat.
_KNOT_FADE = 0.30
_KNOT_FADE_FACE = 0.20

COLLECTION_NAME = "Log Cabin"
TYPE_KEY = "logcabin_type"
WALL_KEY = "logcabin_wall"
COURSE_KEY = "logcabin_course"


def _lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------------------------------
# the log
# --------------------------------------------------------------------------


class Log:
    """A tapered, slightly irregular log lying along a horizontal axis.

    p0 is the butt end (larger radius), p1 the top end. The cuts it carries
    are assigned by generate_cabin from the course schedule:

      groove  - (drop, r_below_at_p0, r_below_at_p1) or None
      notches - list of (distance along axis, drop, radius of crossing log,
                cap) where cap limits the cut to a flat already sawn on the
                crossing member, or None for a full round
      flat_z  - world height to flatten the top to, or None
      flat_span - (from, to) along the axis over which flat_z applies in
                full, or None for the whole length
      flat_shoulders - ((radius, axis height), (radius, axis height)) of the
                logs resting on top at each end of flat_span, whose undersides
                the flat runs out along. None steps straight back to full round
      floor_z - world height to saw the underside flat at, or None
      side_flats - list of (offset along axis, half length, a_min, a_max)
                chiselled pads that truncate the section sideways rather than
                from above or below - a flat seat for a bracket
      tilt    - list of (slope, intercept): planes cutting the top along a
                line across the section, ceiling(a) = a*slope + intercept,
                rather than a height constant across it
      plank   - total height once milled flat top and bottom, centred on the
                axis and local to the log's own frame, or None for round
      cope0, cope1 - (point on axis, direction, radius) of a cylinder to
                scribe the p0/p1 end cap to, or None for a flat cap
      miter   - (world axis, target value): a plumb cut where this log's
                position along that axis reaches target, or None
    """

    def __init__(
        self,
        p0,
        p1,
        r_butt,
        r_top,
        rng,
        bow,
        radial_jitter,
        bow_vertical=0.25,
        knot_density=0.0,
        knot_size=0.06,
        knot_rise=0.008,
    ):
        self.p0 = Vector(p0)
        self.p1 = Vector(p1)
        self.r0 = r_butt
        self.r1 = r_top

        delta = self.p1 - self.p0
        self.length = delta.length
        self.dir = delta.normalized()

        side = self.dir.cross(Vector((0.0, 0.0, 1.0)))
        if side.length < 1e-6:
            side = Vector((1.0, 0.0, 0.0))
        self.side = side.normalized()
        self.up = self.side.cross(self.dir).normalized()

        # A builder rolls the log so its sweep runs sideways, otherwise a bowed
        # log dives into the course below or lifts off it at the ends.
        angle = rng.uniform(0.0, math.tau)
        self.bow_vec = (
            self.side * math.cos(angle)
            + self.up * math.sin(angle) * bow_vertical
        ) * (bow * self.length)

        self.jitter_amp = radial_jitter
        self.jitter_phase = rng.uniform(0.0, math.tau)
        self.jitter_freq = rng.uniform(1.5, 3.5)

        # Knots: the swelling left where a branch grew out of the trunk.
        # Unlike every other feature here they vary with angle as well as
        # position along the log, so they are kept as centres and applied to
        # the barrel rather than folded into the cross-section profile.
        self.knots = []
        expected = knot_density * self.length
        whole = int(expected)
        count = whole + (1 if rng.random() < expected - whole else 0)
        for _ in range(count):
            self.knots.append(
                (
                    rng.uniform(0.02, 0.98),
                    rng.uniform(0.0, math.tau),
                    # A knot of this size covers knot_size of length, and
                    # subtends knot_size / radius around the log.
                    rng.uniform(0.7, 1.4) * knot_size / max(self.length, 1e-6),
                    rng.uniform(0.7, 1.4) * knot_size / max(r_butt, 1e-6),
                    rng.uniform(0.5, 1.0) * knot_rise,
                )
            )

        self.groove = None
        self.notches = []
        self.flat_z = None
        self.flat_span = None
        self.flat_shoulders = None
        self.floor_z = None
        self.side_flats = []
        # (slope, drop at p0, drop at p1): a sloping cut off each end, in the
        # plane of the roof.
        self.rake = None
        # (offset along, radius, axis height) of members let into this one.
        self.pockets = []
        # list of (slope, intercept): planes cutting the top, each a line
        # ceiling(a) = a*slope + intercept across the section, not a height
        # constant across it - a purlin's crown sits tangent-touching the
        # roof plane, but the plane is tilted relative to a circle's own
        # (flat) tangent there, so it cuts through the rest of the barrel
        # rather than grazing past it. Several entries combine by their
        # lowest, which is what gives the ridge its two-sided peak.
        self.tilt = []
        # Total height once milled flat top and bottom, centred on the axis,
        # or None for a plain round log. Local to the log's own frame, so a
        # tilted log keeps a constant thickness along its length rather than
        # a level one.
        self.plank = None
        # (point on axis, direction, radius) of a cylinder to scribe the p0
        # or p1 end cap to, in place of a flat one, or None for a flat cap.
        self.cope0 = None
        self.cope1 = None
        # (world axis, target value): a plane cutting the top where this
        # log's own position along that world axis reaches target - a plumb
        # miter, for two logs meeting the same way from opposite sides of
        # it. Computed fresh from the log's actual axis point at each
        # station rather than baked in once, so it stays exact under bow
        # the way flat_z/plank's own world height would not.
        self.miter = None

    def axis_point(self, t):
        return self.p0.lerp(self.p1, t) + self.bow_vec * math.sin(math.pi * t)

    def knot_offset(self, t, phi):
        """How far the knots push the barrel out at this point on it.

        Squared falloff in both directions, so a knot swells and dies away
        smoothly rather than ending in a rim of its own.
        """
        rise = 0.0
        for centre_t, centre_phi, spread_t, spread_phi, height in self.knots:
            along = (t - centre_t) / spread_t
            if along <= -1.0 or along >= 1.0:
                continue
            around = (phi - centre_phi + math.pi) % math.tau - math.pi
            around /= spread_phi
            if around <= -1.0 or around >= 1.0:
                continue
            fall = (1.0 - along * along) * (1.0 - around * around)
            rise += height * fall * fall
        return rise

    def radius(self, t):
        base = _lerp(self.r0, self.r1, t)
        wobble = 1.0 + self.jitter_amp * math.sin(
            self.jitter_phase + self.jitter_freq * math.tau * t
        )
        return base * wobble


# --------------------------------------------------------------------------
# cross-section profile
# --------------------------------------------------------------------------


# Two outline points closer than this are one vertex. Slots close up onto one
# another whenever the feature between them vanishes, and are welded then.
_WELD = 1e-7

# A floor segment narrower than this is closed up completely rather than left
# as a sliver strip a fraction of a millimetre wide.
_SNAP = 1e-4

# How far a station may round past a notch mouth and still count as on it.
_MOUTH = 1e-5

# Where the first station clear of a notch mouth goes. Just outside, so the
# step down from the notch to the groove reads as the vertical face it is.
_CLEAR = 5e-4

# The thinnest section a raked end is allowed to finish on. Stopping exactly
# where the material runs out would leave a ring of zero area, which doubles
# back on itself and cannot be stitched into a closed shell.
_TIP = 2e-4


def _floor_layout(log, n_up):
    """Vertex slots per half of the floor, as (barrel, flat, groove) counts.

    The underside of a scribed, notched log runs, from each shoulder in to the
    centre line: lower barrel, then the flat of a notch, then the groove. Every
    crease between those is given its own slot, fixed for the whole log, and
    the slots slide continuously as the cuts change along it. A feature that is
    absent at some station simply closes up to zero width and its slots weld.

    Handing a fixed pool of vertices out by width instead - however sensible
    per ring - moves each crease to a different vertex index whenever a break
    appears or disappears, and the quads between two such rings join a crease
    to a non-crease vertex. That is what broke the rim at every corner.

    The counts depend only on which cuts this log carries at all, never on the
    station, so every ring of one log shares a topology.
    """
    span = max(2, n_up - 1)
    # The lower barrel and the groove are swept at the same fixed angular step
    # as the top, with slots enough to reach the bottom of the log. Surplus
    # slots clamp onto a crease and weld, so they cost no vertices.
    barrel = max(2, span // 2)
    # A notch or a sawn floor is a straight line across the section, and its
    # surface is ruled across the log, so one segment per half is exact.
    flat = 1 if (log.notches or log.floor_z is not None) else 0
    groove = max(2, span // 2) if log.groove is not None else 0
    return barrel, flat, groove
def _param_along(log, axis, coord):
    """Where a coordinate falls along a log, as a parameter in [0, 1]."""
    if axis == "X":
        span = log.p1.x - log.p0.x
        start = log.p0.x
    else:
        span = log.p1.y - log.p0.y
        start = log.p0.y
    if abs(span) < 1e-9:
        return 0.0
    return min(max((coord - start) / span, 0.0), 1.0)


def _top_counts(n_up):
    """Slots over the top: (barrel each side, across a cut face)."""
    barrel = max(2, (n_up - 1) // 2)
    # Odd, so one slot sits on the centre line - that is where a raked end
    # parts into its two wings.
    face = 2 * ((barrel + 1) // 2) + 1
    return barrel, face


def _top_points(r, half, ceiling, n_up):
    """The outline over the top of the log, as (a, b, angle) per slot.

    Two regimes share the ring, and each has slots of its own rather than a
    share of one pool. The barrel is swept at fixed angles, the same step the
    rest of the log uses, which is what keeps it round. A cut across the top -
    a sawn flat, or the roof plane at a raked end - is a straight line, and is
    spread across its own width instead: parametrising it by angle on a circle
    it no longer follows sends the outline outside the log as soon as the cut
    passes below the axis, where cos runs out to the full radius.

    Dividing one pool between them left whichever was growing under-described:
    as the roof plane climbs out of the top of the log, the face shrinks to
    nothing while the barrel over it opens out to a quarter circle, and a fixed
    ration of three slots could not follow it. Here each feature keeps its own
    slots and the absent one simply closes up, so the ring costs no more on a
    log with nothing cut off the top than it ever did.

    `angle` is None on the cut itself, where no knot may stand proud.
    """
    barrel_n, face_n = _top_counts(n_up)
    step = math.pi / max(2, n_up - 1)

    # The section's corner on the right, where the sweep hands over to the
    # floor, and the angle it sits at - below the axis once a cut reaches
    # past the log's widest point.
    corner = math.sqrt(max(0.0, r * r - half * half))
    if ceiling is not None:
        corner = min(corner, ceiling)
    phi_corner = math.atan2(corner, half)

    if ceiling is not None and ceiling <= -r:
        # The cut has passed below the whole log: nothing is left of the
        # section here. It closes to a point, which is what tells a raked
        # end where the material has run out.
        return [(0.0, ceiling, None)] * (2 * barrel_n + face_n)

    if ceiling is None or ceiling >= r:
        # Nothing cuts the top: the barrel runs right over the crown, and the
        # face closes onto the crown point.
        rim, face_half, level = 0.5 * math.pi, 0.0, r
    else:
        rim = math.asin(max(-1.0, min(1.0, ceiling / r)))
        face_half = min(half, math.sqrt(max(0.0, r * r - ceiling * ceiling)))
        level = ceiling

    top = max(phi_corner, rim)
    points = []
    for k in range(barrel_n):
        phi = min(max(k * step, phi_corner), top)
        points.append((r * math.cos(phi), r * math.sin(phi), phi))
    for k in range(face_n):
        points.append((face_half * math.cos(math.pi * k / (face_n - 1)), level, None))
    for k in range(barrel_n - 1, -1, -1):
        phi = min(max(k * step, phi_corner), top)
        points.append((-r * math.cos(phi), r * math.sin(phi), math.pi - phi))
    return points


def _ring_points(log, t, along, gap, n_up, layout):
    """The log's outline at one station, as (a, b) in its own frame.

    a runs across the log, b runs up it. Fixed point count, fixed ordering:
    over the top from a=+r to a=-r, then back along the floor. Because the
    topology never changes between stations, rings stitch into clean quads
    with no resampling - which is what stops the surface smearing.
    """
    r = log.radius(t)
    centre = log.axis_point(t)

    # A top flat can be limited to part of the length. A floor beam is sawn
    # flat only between the walls; what projects outside stays a full round,
    # the way an overhanging corner log does.
    ceiling = None
    if log.flat_z is not None:
        top = log.flat_z
        if log.flat_span is not None:
            low, high = log.flat_span
            if along < low or along > high:
                # Past the wall axis the flat runs out along the underside of
                # the log bearing on it, rather than ending in a sawn cliff.
                # The two surfaces stay in contact the whole way, because this
                # follows the very cylinder the log above presents.
                if log.flat_shoulders is None:
                    top = None
                else:
                    near, far = log.flat_shoulders
                    if along < low:
                        reach, (radius, above_z) = low - along, near
                    else:
                        reach, (radius, above_z) = along - high, far
                    if reach >= radius:
                        top = None  # clear of the log above; full round again
                    else:
                        top = max(
                            top, above_z - math.sqrt(radius * radius - reach * reach)
                        )
        # Once the cut would sit above the crown it is not cutting at all.
        if top is not None and top - centre.z < r:
            ceiling = top - centre.z

    # A milled plank: flat top and bottom sawn a fixed distance apart,
    # centred on the *straight* axis, with the round sides left standing.
    # Unlike flat_z/floor_z this is local to the log's own frame rather than
    # a world height, so it stays parallel to the log along its whole run
    # regardless of how the log itself is tilted. "Centred on the straight
    # axis" rather than on centre itself matters wherever bow is in play:
    # up stays fixed along the whole sweep, so bow only ever carries centre
    # off that straight line, never rotates the section - drop is how far
    # off it centre has been carried here, and it is what keeps the two
    # faces on the plane the log was built flush with instead of drifting
    # with whatever centre happens to be at this station.
    if log.plank is not None:
        half_h = log.plank * 0.5
        drop = (centre - log.p0).dot(log.up)
        ceiling = half_h - drop if ceiling is None else min(ceiling, half_h - drop)

    # A gable log is cut off in the plane of the roof at both ends, so the
    # ends line up into one rake instead of stepping. Unlike every other cut
    # this one's height changes along the log; both ends share an expression
    # because the distance to the nearer end is what sets it.
    rake = None
    if log.rake is not None:
        # Each end carries its own drop: the ends differ in radius, so one
        # figure for both overshoots the thin end and drives the section
        # negative there.
        slope, drop_start, drop_end = log.rake
        rake = min(
            log.p0.z - drop_start + along * slope,
            log.p0.z - drop_end + (log.length - along) * slope,
        ) - centre.z
        if rake < r:
            ceiling = rake if ceiling is None else min(ceiling, rake)

    # Seats cut to receive something laid across the log - a purlin let into a
    # gable. This is a saddle notch turned upside down: the crossing member
    # occupies a horizontal slab of the cross-section for exactly the same
    # reason, so the seat follows its cylinder rather than being squared off,
    # and the cut runs from the slab's underside upward because a log has to
    # be laid into an open seat rather than threaded through a hole.
    for offset, radius, axis_z in log.pockets:
        reach = along - offset
        span = radius * radius - reach * reach
        if span <= 0.0:
            continue  # this station clears the crossing
        local = axis_z - math.sqrt(span) - centre.z
        if local < r:
            ceiling = local if ceiling is None else min(ceiling, local)

    # Saddle notches. A perpendicular log removes a horizontal slab, so its
    # contribution to the floor is a constant across the width - not a curve,
    # unlike the groove below - since a crossing log's own length runs the
    # full width and its height there depends only on distance along this
    # log, never on where across it you are.
    line_top = None
    for offset, drop, radius, cap in log.notches:
        reach = radius + gap
        d = abs(along - offset)
        if d > reach + _MOUTH:
            continue  # this station clears the crossing
        # A station placed on the mouth can round a hair either side of it.
        # Inside, the cylinder's wall is vertical and a few micrometres read
        # as millimetres of rise; outside, the notch would vanish outright.
        # Either way it is meant to be the mouth itself.
        d = min(d, reach)
        top = math.sqrt(max(0.0, reach * reach - d * d)) - drop
        # A member that has itself been flattened on top is not a full round,
        # so the notch over it stops at that flat instead of following a
        # cylinder that is no longer there.
        if cap is not None and top > cap:
            top = cap
        # The notch runs its true shape right out to the crossing log's side.
        # At the mouth it sits at the crossing log's axis height, which is
        # also where the groove meets the barrel, so notch, groove and barrel
        # close onto a single point there. The floor's crease slots follow
        # that pinch continuously; no fade is needed to hide a hand-off.
        if line_top is None or top > line_top:
            line_top = top

    # A sill log is sawn flat where it beds on the foundation. That is the same
    # shape as a notch - a horizontal cut - so it folds into line_top and needs
    # no machinery of its own. It also cures the taper float: a round log on a
    # flat foundation only touches where its radius is greatest, at the butt,
    # whereas a sawn flat bears along the whole length.
    if log.floor_z is not None:
        flat = log.floor_z - centre.z
        if line_top is None or flat > line_top:
            line_top = flat

    # The plank's own underside, centred on the straight axis the same way
    # its top is - see the note there on why, and drop is the same figure.
    if log.plank is not None:
        drop = (centre - log.p0).dot(log.up)
        flat = -log.plank * 0.5 - drop
        if line_top is None or flat > line_top:
            line_top = flat

    # A plumb miter: solving centre + up*b for where it crosses the target
    # plane gives the b this station's own material runs out at. At the
    # plane itself that is b=0 - cut clean through the axis into a
    # half-round, not to a point - and on the near side of it (up_component
    # and the side actually being kept share a sign) the log has no reach
    # below that at all; past the plane the other way, nothing here cuts
    # anything, the same as a rake past the end it is not measured from.
    if log.miter is not None:
        axis_vec, target = log.miter
        up_component = log.up.dot(axis_vec)
        if abs(up_component) > 1e-9:
            miter_b = (target - centre.dot(axis_vec)) / up_component
            if line_top is None or miter_b > line_top:
                line_top = miter_b

    # Lateral groove: the arc of the log below, offset down by a full round.
    groove = None
    if log.groove is not None:
        drop, r_below_start, r_below_end = log.groove
        groove = (drop, _lerp(r_below_start, r_below_end, t) + gap)

    def upper(a):
        b = math.sqrt(max(0.0, r * r - a * a))
        return b if ceiling is None else min(b, ceiling)

    def lower(a):
        b = -math.sqrt(max(0.0, r * r - a * a))
        if line_top is not None and line_top > b:
            b = line_top
        if groove is not None:
            drop, r_groove = groove
            if abs(a) < r_groove:
                b = max(b, math.sqrt(r_groove * r_groove - a * a) - drop)
        return min(b, upper(a))

    # The section's half-width, where the sweep hands over to the floor.
    # While every cut stays at or below the axis the material still reaches
    # +/-r. A cut biting above the axis - a saddle notch does, by
    # scribe_depth/2 - takes the shoulders off from below, and a ceiling
    # dropping below the axis takes them off from above.
    half = r
    if line_top is not None and 0.0 < line_top < r:
        half = math.sqrt(r * r - line_top * line_top)
    if ceiling is not None and ceiling < 0.0:
        half = min(half, math.sqrt(max(0.0, r * r - ceiling * ceiling)))
    half = max(half, r * 1e-3)

    # A chiselled pad truncates the section sideways rather than from above or
    # below - the only cut that limits `a` instead of `b`. The vertical face it
    # leaves needs no vertices of its own: slots past it are pinned to it and
    # weld, and the edge from the sweep down to the floor is the pad.
    a_left, a_right = -half, half
    for offset, half_length, low_limit, high_limit in log.side_flats:
        if abs(along - offset) > half_length:
            continue
        a_left = max(a_left, low_limit)
        a_right = min(a_right, high_limit)
    if a_right - a_left < r * 0.05:
        a_left, a_right = -half, half  # a pad that deep would sever the log

    # A tilt cutting deep enough can, before reaching the current shoulder,
    # drop below where the floor's own underside already is - the section
    # has genuinely run out of material there, not just had its top shaved.
    # Clipping the top sweep alone past that point, and leaving a_right/
    # a_left where they were, pulls the top's own edge away from wherever
    # the floor's edge still sits: they are supposed to be the same point.
    # Find where each tilt actually runs out, on whichever side it slopes
    # downhill towards, and pull the shoulder in to meet it.
    def _floor_edge(a):
        return -math.sqrt(max(0.0, r * r - a * a))

    # A tilt's (slope, intercept) is measured against the *straight* axis,
    # the same as a plank's top and bottom are (see the note by log.plank
    # below) - side and up both stay fixed along the whole sweep, so bow
    # only ever carries centre off that line, never rotates the section.
    # Carried sideways, centre drags the a=0 origin the intercept is
    # measured from along with it, which the slope turns into a further
    # height error on top of centre's own vertical drift - up_drift corrects
    # that part directly, side_drift by rejoining the line at the a it
    # would have crossed a=0 at had centre not moved, so tilt below reads
    # the true roof plane rather than one that drifts with whatever centre
    # happens to be at this station.
    up_drift = (centre - log.p0).dot(log.up) if log.tilt else 0.0
    side_drift = (centre - log.p0).dot(log.side) if log.tilt else 0.0

    for slope, intercept in log.tilt:
        intercept += slope * side_drift - up_drift
        if slope < 0.0:
            lo, hi = 0.0, a_right
            if slope * hi + intercept - _floor_edge(hi) < 0.0:
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    if slope * mid + intercept - _floor_edge(mid) >= 0.0:
                        lo = mid
                    else:
                        hi = mid
                a_right = min(a_right, lo)
        elif slope > 0.0:
            lo, hi = a_left, 0.0
            if slope * lo + intercept - _floor_edge(lo) < 0.0:
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    if slope * mid + intercept - _floor_edge(mid) >= 0.0:
                        hi = mid
                    else:
                        lo = mid
                a_left = max(a_left, hi)

    # ---- over the top ------------------------------------------------------
    # The barrel is swept at angles fixed for the whole log, and a shoulder
    # cut clamps the lowest of them onto its rim, where they weld. Spreading
    # the sweep between the rims instead moved every vertex over the crown
    # whenever a notch bit above the axis, bending the long edges down over
    # each corner.
    points = []
    for a, b, phi in _top_points(r, half, ceiling, n_up):
        rise = 0.0
        if log.knots and phi is not None:
            # Knots belong only on surfaces nobody has worked, so they fade
            # out towards the shoulders and are absent from a cut face.
            reach = math.atan2(
                min(math.sqrt(max(0.0, r * r - half * half)), ceiling)
                if ceiling is not None
                else math.sqrt(max(0.0, r * r - half * half)),
                half,
            )
            mask = min(1.0, min(phi - reach, math.pi - reach - phi) / _KNOT_FADE)
            # Near a notch the surviving web narrows at a rate set by the
            # notch, not the knot, so bring the knot down with the cut.
            mask *= (half / r) ** 2
            if ceiling is not None:
                mask = min(mask, (ceiling - b) / (r * _KNOT_FADE_FACE))
            if mask > 0.0:
                rise = log.knot_offset(t, phi) * min(1.0, mask)
        if rise:
            a, b = a + rise * math.cos(phi), b + rise * math.sin(phi)
            if ceiling is not None:
                b = min(b, ceiling)
        # A chiselled pad truncates the section sideways; slots past it are
        # pinned to its face.
        if a > a_right:
            a, b = a_right, min(b, upper(a_right))
        elif a < a_left:
            a, b = a_left, min(b, upper(a_left))
        # A tilt plane only ever trims the top: it is a straight line with no
        # curvature, so past the shoulder it keeps descending long after the
        # barrel has curved back up to meet the floor, and applying it there
        # would carve into the underside instead of stopping at the crown.
        if log.tilt:
            b = min(
                b,
                min(s * a + i + s * side_drift - up_drift for s, i in log.tilt),
            )
        points.append((a, b))

    # ---- along the floor ---------------------------------------------------
    # Creases on the right half, measured out from the centre line:
    #   0 <= edge <= rim <= half
    # rim  - where the barrel stops and a cut begins (notch or groove);
    # edge - where the notch's flat hands over to the groove's crescent.
    # The groove is highest at the centre and the barrel lowest there, so a
    # flat line meets each of them at most once per side and the order holds.
    barrel_n, flat_n, groove_n = layout

    g_rim = 0.0
    drop_g = r_groove = 0.0
    if groove is not None:
        drop_g, r_groove = groove
        g_rim = None
        if drop_g > 1e-9:
            # Circle meets circle: with A = sqrt(rg^2 - u), B = sqrt(r^2 - u)
            # and A + B = drop, then A - B = (rg^2 - r^2)/drop.
            arc = (drop_g + (r_groove * r_groove - r * r) / drop_g) * 0.5
            u = r_groove * r_groove - arc * arc
            if u > 0.0:
                g_rim = math.sqrt(u)
        if g_rim is None:
            # The circles do not cross: the groove either misses the log or
            # takes its whole underside.
            g_rim = min(r, r_groove) if r_groove - drop_g > -r else 0.0

    n_rim = 0.0
    if line_top is not None and -r < line_top < r:
        n_rim = math.sqrt(r * r - line_top * line_top)

    rim = min(half, max(n_rim, g_rim))
    if groove is None:
        edge = 0.0  # the flat, if any, runs right across
    elif line_top is None:
        edge = rim  # no flat at this station
    else:
        height = line_top + drop_g
        if height >= r_groove:
            edge = 0.0  # the notch clears the groove's crown entirely
        elif height <= 0.0:
            edge = rim  # the notch sits wholly below the groove
        else:
            edge = min(rim, math.sqrt(r_groove * r_groove - height * height))

    # Close up slivers rather than leave strips a hair wide.
    if rim < _SNAP:
        rim = 0.0
    if rim - edge < _SNAP:
        edge = rim
    if edge < _SNAP:
        edge = 0.0

    # Lower barrel, by angle from the shoulder down to the rim.
    if half >= r * (1.0 - 1e-9):
        phi_s = 0.0
    elif line_top is not None and line_top > 0.0:
        phi_s = math.atan2(line_top, half)
    else:
        phi_s = -math.atan2(math.sqrt(max(0.0, r * r - half * half)), half)
    b_rim = lower(rim)
    if rim > 1e-12:
        phi_r = math.atan2(b_rim, rim)
    else:
        phi_r = -0.5 * math.pi if b_rim < 0.0 else 0.5 * math.pi
    phi_r = min(phi_r, phi_s)

    # Fixed angles again, clamped onto the creases, for the same reason as
    # over the top: re-spreading the slots between the shoulder and the rim
    # swung every one of them sideways whenever the rim moved, which bent the
    # long edges along each notch and, where the rim jumps at a notch mouth,
    # left strips of skewed faces shading as if they faced along the log.
    step = math.pi / max(2, n_up - 1)
    snap_r = _SNAP / max(r, 1e-9)

    run = []  # right half, from the shoulder in to the centre line
    for k in range(1, barrel_n):
        phi = -k * step
        if k == 1 and phi_s == 0.0 and log.tilt:
            # phi_s = 0 exactly wherever nothing narrows the section, which
            # is the ordinary state along most of an untouched log's length
            # too - restricting this to log.tilt keeps it off logs that ever
            # carry a notch or groove, where phi_s flips to nonzero at the
            # edge of that feature's reach and pinning here would open a new
            # step right at that boundary instead of closing this one. A
            # tilted log (a purlin, the ridge) never carries either, so
            # phi_s stays 0 along its whole length and there is no boundary
            # to create. The top's own barrel sweep starts right at the
            # shoulder (its k=0), but this loop starts at k=1, so its first
            # slot lands one whole step short - invisible on a plain round
            # log, but a visible stray edge against a flat cut.
            run.append((r, 0.0))
            continue
        if phi >= phi_s - snap_r:
            if run:
                run.append(run[-1])
            else:
                # phi_s marks the shoulder on the log's own natural circle -
                # right for a notch or groove, which never cuts above the
                # axis by more than a sliver. A steep-enough ceiling collapses
                # that shoulder toward the centre line without moving phi_s
                # to match, so the point this angle names can sit above the
                # ceiling that is supposed to be the highest thing left.
                a0 = r * math.cos(phi_s)
                b0 = r * math.sin(phi_s)
                if ceiling is not None:
                    b0 = min(b0, ceiling)
                run.append((a0, b0))
            continue
        if phi <= phi_r + snap_r:
            run.append((rim, b_rim))
            continue
        a = r * math.cos(phi)
        run.append((a, min(r * math.sin(phi), upper(a))))
    run.append((rim, b_rim))

    for k in range(1, flat_n + 1):
        a = _lerp(rim, edge, k / flat_n)
        run.append((a, lower(a)))

    if groove_n:
        alpha_edge = (
            math.asin(min(1.0, edge / r_groove)) if r_groove > 1e-9 else 0.0
        )
        snap_g = _SNAP / max(r_groove, 1e-9)
        for m in range(groove_n - 1, -1, -1):
            alpha = m * step
            if alpha >= alpha_edge - snap_g:
                a = edge
            else:
                a = r_groove * math.sin(alpha)
            run.append((a, lower(a)))

    def seat(a, b):
        if a > a_right:
            return a_right, lower(a_right)
        if a < a_left:
            return a_left, lower(a_left)
        return a, b

    # Ring order: over the top from +a to -a, then back along the floor.
    for a, b in run:
        points.append(seat(-a, b))
    for a, b in reversed(run[:-1]):
        points.append(seat(a, b))

    # Towards a raked end the roof plane drops below the crown of the groove,
    # and from there the section is two wings, one either side of the groove,
    # with nothing between them: the log below fills that space right up to
    # the same roof plane. `web` is the half-width of that gap. One outline
    # can only bridge it with a zero-thickness sheet, so build_log_mesh parts
    # the ring into its two halves wherever it is open.
    web = 0.0
    if (
        rake is not None
        and groove is not None
        and ceiling is not None
        and abs(ceiling - rake) < 1e-12
    ):
        height = ceiling + drop_g
        if 0.0 < height < r_groove:
            web = min(half, math.sqrt(r_groove * r_groove - height * height))
        if web < _SNAP:
            web = 0.0

    # Which crease slots are live creases at this station: the rim wherever
    # anything cuts the underside, the edge only while notch and groove are
    # both actually showing.
    return points, (rim > 0.0, 0.0 < edge < rim), web


def _rake_ends(log, gap, n_up):
    """Where a grooved, raked log really ends, and where it parts in two.

    Returns (tip_start, part_start, part_end, tip_end) as distances along the
    axis, or None. The rake is set to close a round log at the bottom of its
    barrel, but the groove has already taken that away - and a notch near the
    end can take away more - so what is left ends short of that, wherever the
    section first has any thickness. That is measured from the section itself
    rather than predicted, so every cut is accounted for. Towards the end,
    from where the roof plane drops below the groove's crown, the log is two
    wings; part_start and part_end are where that begins.
    """
    if log.rake is None or log.groove is None or log.length <= 1e-9:
        return None
    slope, drop_start, drop_end = log.rake
    drop, r_below_start, r_below_end = log.groove
    if drop <= 1e-9:
        return None
    length = log.length
    layout = _floor_layout(log, n_up)

    def extent(along):
        """Height of the section - its thickest point - at this distance."""
        t = min(max(along / length, 0.0), 1.0)
        points = _ring_points(log, t, along, gap, n_up, layout)[0]
        heights = [b for _a, b in points]
        return max(heights) - min(heights)

    def below_crown(along):
        """How far the roof plane sits above the groove's crown here."""
        t = min(max(along / length, 0.0), 1.0)
        plane = min(
            log.p0.z - drop_start + along * slope,
            log.p0.z - drop_end + (length - along) * slope,
        ) - log.axis_point(t).z
        r_groove = _lerp(r_below_start, r_below_end, t) + gap
        return plane - (r_groove - drop)

    def bisect(test, outer, inner):
        """Where test(along) turns positive, between the end and inside."""
        for _ in range(48):
            middle = 0.5 * (outer + inner)
            if test(middle) > 0.0:
                inner = middle
            else:
                outer = middle
        return inner

    middle = 0.5 * length
    step = 0.002

    def tip(outer):
        """First point in from this end with real material."""
        inward = 1.0 if outer < middle else -1.0
        if extent(outer) > _TIP:
            return None  # nothing wasted at this end
        count = int(abs(middle - outer) / step)
        for k in range(1, count + 1):
            along = outer + inward * step * k
            if extent(along) > _TIP:
                return bisect(
                    lambda x: extent(x) - _TIP, along - inward * step, along
                )
        return None

    def part(outer, inner):
        if below_crown(outer) >= 0.0 or below_crown(inner) <= 0.0:
            return None
        return bisect(below_crown, outer, inner)

    tip_start = tip(0.0)
    tip_end = tip(length)
    if tip_start is None or tip_end is None or tip_start >= tip_end:
        return None
    part_start = part(0.0, middle)
    part_end = part(length, middle)
    if part_start is None or part_start <= tip_start:
        part_start = tip_start
    if part_end is None or part_end >= tip_end:
        part_end = tip_end
    return tip_start, part_start, part_end, tip_end


def _station_params(log, axial, refine, gap=0.0, n_up=15):
    """Uniform stations, plus a dense cluster at each notch.

    A notch spans about one log diameter out of several metres. Sampling the
    whole log finely enough to catch it would be wasted everywhere else, and
    sampling it coarsely is what made notches look chiselled.
    """
    params = {round(i / axial, 6) for i in range(axial + 1)}
    if log.length > 1e-9:
        # The rake runs out over about a radius' worth of length at each end,
        # which the uniform stations would cross in a single step.
        if log.rake is not None:
            slope, _drop_start, _drop_end = log.rake
            if slope > 1e-6:
                reach = min(log.length * 0.5, 2.5 * log.radius(0.0) / slope)
                for k in range(refine + 1):
                    near = reach * k / refine
                    for edge in (near, log.length - near):
                        param = edge / log.length
                        if 0.0 <= param <= 1.0:
                            params.add(round(param, 6))

        # A miter closes over about a radius' worth of length too, the same
        # reason a rake does, just centred on wherever the axis actually
        # crosses the target plane rather than pinned to either end.
        if log.miter is not None:
            axis_vec, target = log.miter
            rate = log.dir.dot(axis_vec)
            if abs(rate) > 1e-6:
                cross_t = (target - log.p0.dot(axis_vec)) / (rate * log.length)
                reach = min(0.5, 2.5 * log.radius(0.0) / (abs(rate) * log.length))
                for k in range(refine + 1):
                    offset = reach * k / refine
                    for param in (cross_t - offset, cross_t + offset):
                        if 0.0 <= param <= 1.0:
                            params.add(round(param, 6))

        # A seat curves like the member it receives, so it is sampled by the
        # crossing angle for the same reason a notch is: that gathers stations
        # at the mouth, where sqrt(R^2 - e^2) turns hardest.
        for offset, radius, _axis_z in log.pockets:
            for k in range(refine + 1):
                psi = math.pi * (k / refine - 0.5)
                param = (offset + radius * math.sin(psi)) / log.length
                if 0.0 <= param <= 1.0:
                    params.add(round(param, 6))
            for edge, outward in ((offset - radius, -1.0), (offset + radius, 1.0)):
                param = (edge + outward * 0.004) / log.length
                if 0.0 <= param <= 1.0:
                    params.add(round(param, 6))

        # A knot is a few centimetres on a log several metres long, so the
        # uniform stations will step straight over it. Each one gets its own.
        for centre_t, _phi, spread_t, _spread_phi, _rise in log.knots:
            for step in range(-3, 4):
                param = centre_t + spread_t * step / 3.0
                if 0.0 <= param <= 1.0:
                    params.add(round(param, 6))

        for offset, _drop, radius, _cap in log.notches:
            # Sampled against the cut's own reach, gap included, so the last
            # station lands exactly where the notch meets the crossing log's
            # side rather than a millimetre inside it, up the steep wall.
            radius += gap
            # The notch floor is sqrt(R^2 - e^2), whose slope runs away at the
            # mouth. Sampling evenly along the log therefore spends stations
            # on the crown, which is nearly flat, and starves the mouth, which
            # is where the surface actually turns. Stepping by the crossing
            # angle instead - e = R sin(psi) - spaces stations evenly around
            # the cylinder being cut, so they gather where the curvature is
            # and land exactly on the mouth at both ends.
            for k in range(refine + 1):
                psi = math.pi * (k / refine - 0.5)
                param = (offset + radius * math.sin(psi)) / log.length
                if 0.0 <= param <= 1.0:
                    params.add(round(param, 6))

            # One station just clear of each mouth. The crossing log's surface
            # sits at its own axis height right where it runs out, which is
            # generally not where the groove meets the barrel - a butt resting
            # on a top is fatter than the groove cut for it - so there is a
            # genuine vertical step down to the groove there. Keeping this
            # station within a hair of the mouth makes that step a clean
            # face instead of a ramp.
            for edge, outward in ((offset - radius, -1.0), (offset + radius, 1.0)):
                param = (edge + outward * _CLEAR) / log.length
                if 0.0 <= param <= 1.0:
                    params.add(round(param, 6))

        # Where a top flat starts and stops, the ring's topology changes: two
        # of its vertices become pinned to the flat's rim. Pinning stations
        # either side confines that change to a few millimetres so it reads as
        # an edge rather than a smeared, torn strip.
        if log.flat_span is not None:
            for edge in log.flat_span:
                for nudge in (-0.002, 0.002):
                    param = (edge + nudge) / log.length
                    if 0.0 <= param <= 1.0:
                        params.add(round(param, 6))

            # The flat runs out along the underside of the log above, so that
            # stretch needs its own stations. Stepping by angle gathers them
            # towards the outer end, where sqrt(R^2 - d^2) turns hardest.
            if log.flat_shoulders is not None:
                low, high = log.flat_span
                near, far = log.flat_shoulders
                ends = ((low, near, -1.0), (high, far, 1.0))
                for edge, (radius, above_z), outward in ends:
                    for k in range(refine + 1):
                        reach = radius * math.sin(math.pi * 0.5 * k / refine)
                        param = (edge + outward * reach) / log.length
                        if 0.0 <= param <= 1.0:
                            params.add(round(param, 6))

                    # And one where that curve lifts clear of the flat, which
                    # is a crease between two different surfaces.
                    rise = above_z - log.flat_z
                    if 0.0 < rise < radius:
                        reach = math.sqrt(radius * radius - rise * rise)
                        param = (edge + outward * reach) / log.length
                        if 0.0 <= param <= 1.0:
                            params.add(round(param, 6))

    # A grooved gable log's material stops short of where the rake would
    # close a full round, so nothing is built past that: what used to be
    # built there was a sheet of zero thickness lying in the roof plane. The
    # stretch where the log runs out as two wings gets stations of its own,
    # and the stations where it parts and where it ends must land exactly.
    first, last = 0.0, 1.0
    pinned = set()
    ends = _rake_ends(log, gap, n_up)
    if ends is not None:
        tip_start, part_start, part_end, tip_end = ends
        first, last = tip_start / log.length, tip_end / log.length
        params = {p for p in params if first < p < last}
        steps = max(4, refine // 2)
        for tip, part in ((tip_start, part_start), (tip_end, part_end)):
            for k in range(1, steps):
                params.add(round(_lerp(tip, part, k / steps) / log.length, 6))
        pinned = {first, last, part_start / log.length, part_end / log.length}
        params |= pinned

    # Rounding alone can still leave stations a micrometre apart, and a ring
    # pair that close makes zero-area quads that shade black. Where a pinned
    # station crowds an ordinary one, the pinned one stays.
    ordered = sorted(p for p in params if first <= p <= last)
    spaced = [ordered[0]]
    for value in ordered[1:]:
        if value - spaced[-1] > 1e-5:
            spaced.append(value)
        elif value in pinned and spaced[-1] not in pinned:
            spaced[-1] = value
    if spaced[-1] < last - 1e-9:
        spaced[-1] = last
    return spaced


def build_log_mesh(log, axial, radial, gap, refine, creases=None):
    """Return (verts, faces) for one finished log, in world coordinates.

    log.cope0/log.cope1, if set, replace the flat cap at the p0/p1 end with
    one scribed to a crossing cylinder instead - see _cope_cap. Each is
    (point on the target's axis, its direction, its radius).

    Every ring has the same slot count. Slots that coincide - because the
    feature between them has closed up at that station - share one vertex,
    and a quad that loses a corner that way becomes a triangle, or goes if it
    loses two. That is what lets a crease keep its slot along the whole log.

    Where a raked end parts into two wings (see _ring_points' `web`), each
    half of the ring is carried on as a tube of its own. The two start from
    the pinch where the roof plane just touches the groove's crown and taper
    to a point where the material ends, so the shell stays closed.

    If `creases` is a list, the vertex pairs of every edge running along a
    cut's crease are appended to it. Those are known exactly here, whereas an
    angle threshold misses the shallow ones - where a notch hands over to the
    groove - and smooth shading then smears one cut's normals over the other.
    """
    n_up = max(3, radial // 2 + 1)
    if n_up % 2 == 0:
        # An odd count puts a slot on the crown line, which is where a raked
        # end parts into its two halves.
        n_up += 1
    layout = _floor_layout(log, n_up)
    params = _station_params(log, axial, refine, gap, n_up)

    barrel_n, flat_n, groove_n = layout
    run = barrel_n + flat_n + groove_n
    top_barrel, top_face = _top_counts(n_up)
    top_n = 2 * top_barrel + top_face
    size = top_n + 2 * run - 1
    mid = top_barrel + (top_face - 1) // 2
    centre_slot = top_n + run - 1
    # Each half keeps the ring's own winding, so the two tubes face outward.
    right_slots = list(range(0, mid + 1)) + list(range(centre_slot, size))
    left_slots = list(range(mid, top_n)) + list(range(top_n, centre_slot + 1))

    def close(p, q):
        return abs(p[0] - q[0]) <= _WELD and abs(p[1] - q[1]) <= _WELD

    verts = []

    def place(centre, ring, pinch=None):
        """Vertex ids for an outline, welding coincident neighbours."""
        count = len(ring)
        owner = list(range(count))
        for j in range(1, count):
            if close(ring[j], ring[j - 1]):
                owner[j] = owner[j - 1]
        if count > 1 and close(ring[-1], ring[0]) and owner[-1] != owner[0]:
            tail = owner[-1]
            owner = [owner[0] if o == tail else o for o in owner]
        if pinch is not None:
            # The crown slot over the top and the centre slot of the floor
            # meet where the ring is about to part. They are not neighbours
            # in the ring, but they are one point.
            i, j = pinch
            if close(ring[i], ring[j]) and owner[i] != owner[j]:
                gone = owner[j]
                owner = [owner[i] if o == gone else o for o in owner]
        index = {}
        ids = []
        for j in range(count):
            o = owner[j]
            if o not in index:
                index[o] = len(verts)
                a, b = ring[o]
                verts.append(centre + log.side * a + log.up * b)
            ids.append(index[o])
        return ids

    sections = []
    for t in params:
        ring, live, web = _ring_points(log, t, t * log.length, gap, n_up, layout)
        sections.append((log.axis_point(t), ring, live, web))

    stations = []  # (whole ring ids or None, right half ids, left half ids)
    flags = []
    for i, (centre, ring, live, web) in enumerate(sections):
        flags.append(live)
        if web <= 0.0:
            # Only where the ring is about to part, or has just closed up,
            # are its crown and floor centre one point. Elsewhere a section
            # pinched to nothing at the centre - a seat cut down to the
            # groove - keeps two vertices, or the shell pinches shut there.
            parting = (i > 0 and sections[i - 1][3] > 0.0) or (
                i + 1 < len(sections) and sections[i + 1][3] > 0.0
            )
            ids = place(centre, ring, (mid, centre_slot) if parting else None)
            stations.append(
                (ids, [ids[s] for s in right_slots], [ids[s] for s in left_slots])
            )
            continue
        # Open across the middle: every slot that falls in the gap is pinned
        # to the gap's edge, on the roof plane, and the halves build apart.
        plane = ring[mid][1]
        right = []
        for s in right_slots:
            a, b = ring[s]
            right.append((web, plane) if a < web else (a, b))
        left = []
        for s in left_slots:
            a, b = ring[s]
            left.append((-web, plane) if a > -web else (a, b))
        stations.append((None, place(centre, right), place(centre, left)))

    def loop_of(corners):
        loop = []
        for v in corners:
            if not loop or loop[-1] != v:
                loop.append(v)
        if len(loop) > 1 and loop[-1] == loop[0]:
            loop.pop()
        return loop

    pairs = list(zip(stations, stations[1:]))

    if creases is not None:
        rim_slots = (top_n + barrel_n - 1, top_n + 2 * run - 1 - barrel_n)
        edge_slots = (
            (top_n + barrel_n - 1 + flat_n, top_n + 2 * run - 1 - barrel_n - flat_n)
            if flat_n and groove_n
            else ()
        )

        def slot_id(station, slot):
            whole, right, left = station
            if whole is not None:
                return whole[slot]
            if slot in right_slots:
                return right[right_slots.index(slot)]
            return left[left_slots.index(slot)]

        for i, (low, high) in enumerate(pairs):
            for kind, slots in ((0, rim_slots), (1, edge_slots)):
                if not (flags[i][kind] and flags[i + 1][kind]):
                    continue
                for slot in slots:
                    here, there = slot_id(low, slot), slot_id(high, slot)
                    if here != there:
                        creases.append((here, there))

    faces = []

    def band(low, high):
        count = len(low)
        for j in range(count):
            j_next = (j + 1) % count
            face = loop_of((low[j], high[j], high[j_next], low[j_next]))
            if len(face) >= 3:
                faces.append(tuple(face))

    for low, high in pairs:
        if low[0] is not None and high[0] is not None:
            band(low[0], high[0])
        else:
            band(low[1], high[1])
            band(low[2], high[2])

    # Cap centres use the ring centroid, not the axis: a deeply notched end
    # ring can leave the axis outside the remaining outline. A parted end is
    # capped half by half, and a half that has closed to a point needs none.
    for station, outward, cope in (
        (stations[0], False, log.cope0),
        (stations[-1], True, log.cope1),
    ):
        whole, right, left = station
        outlines = [whole] if whole is not None else [right, left]
        for ids in outlines:
            loop = loop_of(ids)
            if len(loop) < 3:
                continue
            if cope is not None:
                _cope_cap(verts, faces, log, loop, outward, cope)
                continue
            total = Vector((0.0, 0.0, 0.0))
            for v in loop:
                total += verts[v]
            centre_index = len(verts)
            verts.append(total / len(loop))
            for k in range(len(loop)):
                here, after = loop[k], loop[(k + 1) % len(loop)]
                if outward:
                    faces.append((centre_index, here, after))
                else:
                    faces.append((centre_index, after, here))

    return verts, faces


def _cope_push(p, direction, target, outward):
    """Where a point on a straight run - along `direction` - first meets a
    crossing cylinder: a line-vs-cylinder solve, exact rather than
    approximated, since it is the point's own line that is moving.

    `target` is (point on the cylinder's axis, its direction, its radius).
    `outward` picks which of the line's two crossings with it is wanted -
    the one further on from p in `direction`, or the one further back
    against it - since the nearer crossing in that direction is the first
    surface actually met, not the far side of it.
    """
    r0, axis_dir, radius = target
    k = direction.dot(axis_dir)
    offset = p - r0
    along_dir = offset.dot(direction)
    along_axis = offset.dot(axis_dir)
    coeff_a = 1.0 - k * k
    coeff_b = 2.0 * (along_dir - k * along_axis)
    coeff_c = offset.length_squared - along_axis * along_axis - radius * radius
    if abs(coeff_a) < 1e-9:
        t = -coeff_c / coeff_b if abs(coeff_b) > 1e-9 else 0.0
    else:
        disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c
        # The closest approach - the midpoint between the two crossings, or
        # where they would be were they real - is the fallback for a point
        # whose own line passes clear of the cylinder altogether, or only
        # crosses it on the wrong side. Typically a board corner that
        # overhangs past a round log's shoulder: there is no true crossing
        # in the direction wanted, so nudging it as close as the geometry
        # allows still closes most of the gap, rather than leaving that
        # corner at its unmoved, uncoped length.
        closest = -coeff_b / (2.0 * coeff_a)
        if disc < 0.0:
            t = closest
        else:
            root = math.sqrt(disc)
            t1 = (-coeff_b - root) / (2.0 * coeff_a)
            t2 = (-coeff_b + root) / (2.0 * coeff_a)
            if outward:
                candidates = [x for x in (t1, t2) if x >= 0.0]
                t = min(candidates) if candidates else closest
            else:
                candidates = [x for x in (t1, t2) if x <= 0.0]
                t = max(candidates) if candidates else closest
    return p + direction * t


def _cope_cap(verts, faces, log, loop, outward, target):
    """Replace a flat end cap with one scribed to a crossing cylinder.

    Every other cut in this file varies a cross-section's shape along the
    log's length; this is the one place a log's own length varies instead,
    per point around its cross-section - so it is built as a one-off patch
    rather than folded into _ring_points' station sweep.

    Each vertex already on the end ring is pushed out along the log's own
    axis until it lands on `target`'s surface (_cope_push). The pushed-out
    ring is not planar in general, so it gets its own centroid fan rather
    than reusing a flat one.
    """
    dir_ = log.dir
    pushed = []
    for idx in loop:
        new_id = len(verts)
        verts.append(_cope_push(verts[idx], dir_, target, outward))
        pushed.append(new_id)

    # Same band winding every barrel quad in this file uses: (low, high,
    # high_next, low_next). The pushed ring sits further along dir_ than the
    # ring it grew from, so it is "high" facing out, "low" facing back.
    count = len(pushed)
    for j in range(count):
        j_next = (j + 1) % count
        if outward:
            faces.append((loop[j], pushed[j], pushed[j_next], loop[j_next]))
        else:
            faces.append((pushed[j], loop[j], loop[j_next], pushed[j_next]))

    total = Vector((0.0, 0.0, 0.0))
    for v in pushed:
        total += verts[v]
    centre_index = len(verts)
    verts.append(total / count)
    for j in range(count):
        here, after = pushed[j], pushed[(j + 1) % count]
        if outward:
            faces.append((centre_index, here, after))
        else:
            faces.append((centre_index, after, here))


def build_board_mesh(
    point,
    direction,
    a0,
    a1,
    cope0,
    cope1,
    outward0=False,
    outward1=True,
    height_segments=1,
):
    """An oriented board between along=a0 and a1, each end optionally
    scribed to a crossing cylinder (_cope_push) - not just at its four
    corners, but in `height_segments` steps up each vertical edge, so the
    cut traces the round member's curve rather than cutting a straight
    diagonal across it. height_segments=1 is the plain four-corner box.

    `point(along, thick, height)` places a point in the board's own frame -
    thick 0/1 for near/far face, height 0..1 up it - which is what lets
    this serve a board on a roof plane, a floor, or anywhere else without
    knowing that frame itself.

    outward0/outward1 tell each cope which of the target's two crossings to
    use. The default suits a log-like tip extending out to meet something
    beyond its own nominal end - out past a0 against `direction`, on past
    a1 with it. A board whose a0/a1 already sit *at* what it is scribed to,
    such as a noggin nominally spanning centre to centre of the rafters
    either side of it, wants the opposite: pulled in from each end instead
    of reached out from it, so the caller passes outward0=True, outward1=
    False there.
    """
    k = max(1, height_segments)
    heights = [i / k for i in range(k + 1)]

    def chain(along, thick, cope, out):
        pts = []
        for h in heights:
            p = point(along, thick, h)
            if cope is not None:
                p = _cope_push(p, direction, cope, out)
            pts.append(p)
        return pts

    verts = []

    def add(pts):
        base = len(verts)
        verts.extend(pts)
        return base

    # Four chains, back/front at each end, each height_segments+1 points up
    # it. A coped chain bows toward the target rather than staying straight,
    # so the back and front faces below are strips along it, not flat quads,
    # and the two end caps are a direct ladder between back and front rather
    # than a fan - the coped outline is concave, wrapped part way round the
    # target's curve, and a fan from a single centre crosses itself on a
    # shape like that.
    nb = add(chain(a0, 0.0, cope0, outward0))
    nf = add(chain(a0, 1.0, cope0, outward0))
    fb = add(chain(a1, 0.0, cope1, outward1))
    ff = add(chain(a1, 1.0, cope1, outward1))

    faces = [
        (nb, nf, ff, fb),  # bottom
        (nb + k, fb + k, ff + k, nf + k),  # top
    ]
    for i in range(k):
        faces.append((nb + i, fb + i, fb + i + 1, nb + i + 1))  # back
        faces.append((ff + i, nf + i, nf + i + 1, ff + i + 1))  # front
        faces.append((fb + i, ff + i, ff + i + 1, fb + i + 1))  # far cap
        faces.append((nf + i, nb + i, nb + i + 1, nf + i + 1))  # near cap
    return verts, faces


def build_box_mesh(lo, hi):
    """Axis-aligned box from two opposite corners. Sawn timber and brackets."""
    verts = [
        Vector((lo.x, lo.y, lo.z)),
        Vector((hi.x, lo.y, lo.z)),
        Vector((hi.x, hi.y, lo.z)),
        Vector((lo.x, hi.y, lo.z)),
        Vector((lo.x, lo.y, hi.z)),
        Vector((hi.x, lo.y, hi.z)),
        Vector((hi.x, hi.y, hi.z)),
        Vector((lo.x, hi.y, hi.z)),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return verts, faces


def build_slab_mesh(corners, normal, thickness, top=None):
    """Oriented box: a planar quad extruded along its own normal.

    `corners` run the same way build_box_mesh's own lo-face does - along one
    edge, then the other - so a tilted sheet uses the identical face winding
    a level one does, just built from four given points instead of two axis-
    aligned ones. `top`, if given, places the far face's four corners
    directly, for an end cut on a slant instead of square to the sheet.
    """
    lo = list(corners)
    hi = list(top) if top is not None else [c + normal * thickness for c in lo]
    verts = lo + hi
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return verts, faces


def build_grid_slab(bot, top, skip=(), flip=False):
    """A slab over a grid of cells, some of which may be left open.

    `bot` and `top` are matching [row][column] grids of points, the near and
    far faces. Every cell not in `skip` gets a face on each, and a wall
    wherever it borders an open cell or the outside, so a cell left open is a
    clean through-hole. Faces wind as build_slab_mesh's do; `flip` reverses
    them for a slab built on the mirrored side of a roof.
    """
    rows, cols = len(bot), len(bot[0])
    skip = set(skip)
    verts = [p for row in bot for p in row] + [p for row in top for p in row]
    offset = rows * cols

    def b(i, j):
        return i * cols + j

    def t(i, j):
        return offset + i * cols + j

    def solid(i, j):
        return 0 <= i < rows - 1 and 0 <= j < cols - 1 and (i, j) not in skip

    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            if not solid(i, j):
                continue
            faces.append((t(i, j), t(i + 1, j), t(i + 1, j + 1), t(i, j + 1)))
            faces.append((b(i, j), b(i, j + 1), b(i + 1, j + 1), b(i + 1, j)))
            if not solid(i, j - 1):
                faces.append((b(i, j), b(i + 1, j), t(i + 1, j), t(i, j)))
            if not solid(i + 1, j):
                faces.append((b(i + 1, j), b(i + 1, j + 1), t(i + 1, j + 1), t(i + 1, j)))
            if not solid(i, j + 1):
                faces.append((b(i + 1, j + 1), b(i, j + 1), t(i, j + 1), t(i + 1, j + 1)))
            if not solid(i - 1, j):
                faces.append((b(i, j + 1), b(i, j), t(i, j), t(i, j + 1)))
    if flip:
        faces = [tuple(reversed(f)) for f in faces]
    return verts, faces


def merge_meshes(meshes):
    """One (verts, faces) from several, each keeping its own shell."""
    verts, faces = [], []
    for v, f in meshes:
        base = len(verts)
        verts.extend(v)
        faces.extend(tuple(i + base for i in face) for face in f)
    return verts, faces


def build_corrugated_slab(along0, along1, d0, d1, point, normal, lift, thickness, pitch, depth):
    """A ribbed slab: `point(along, d)` is the flat roof plane's own point,
    `normal` its perpendicular. The ribs run down the slope - along `d`,
    constant there - so only `along` needs to be sampled finely; `d` only
    ever needs its two ends, the same as build_slab_mesh's own corners.
    """
    span = along1 - along0
    count = max(2, math.ceil(span / (pitch / 8.0)) + 1)
    alongs = [along0 + span * i / (count - 1) for i in range(count)]

    def rise(a):
        return lift + depth * math.sin(2.0 * math.pi * (a - along0) / pitch)

    top0, top1, bot0, bot1 = [], [], [], []
    for a in alongs:
        base0, base1 = point(a, d0), point(a, d1)
        h = rise(a)
        top0.append(base0 + normal * h)
        top1.append(base1 + normal * h)
        bot0.append(base0 + normal * (h - thickness))
        bot1.append(base1 + normal * (h - thickness))
    verts = top0 + top1 + bot0 + bot1
    n = count

    faces = []
    for i in range(n - 1):
        j = i + 1
        faces.append((i, n + i, n + j, j))  # top, facing +normal
        faces.append((2 * n + j, 3 * n + j, 3 * n + i, 2 * n + i))  # bottom
        faces.append((2 * n + i, 2 * n + j, j, i))  # d0 edge
        faces.append((n + i, n + j, 3 * n + j, 3 * n + i))  # d1 edge
    faces.append((0, n, 3 * n, 2 * n))  # along0 end cap
    last = n - 1
    faces.append((last, 2 * n + last, 3 * n + last, n + last))  # along1 end cap
    return verts, faces


def _frame_point(along_y):
    """Map (along, across, z) into world space, for members running one way."""
    if along_y:
        return lambda along, across, z: Vector((across, along, z))
    return lambda along, across, z: Vector((along, across, z))


def _frame_place(along_y):
    """The same mapping, for the two opposite corners of a box."""
    put = _frame_point(along_y)

    def place(along0, along1, across0, across1, z0, z1):
        return put(along0, across0, z0), put(along1, across1, z1)

    return place


def _even_run(first, last, pitch):
    """Member centres from first to last inclusive, on pitch where it fits.

    The two end bays take up whatever does not divide evenly. They are made as
    large as the pitch allows rather than as small as possible: packing in the
    maximum number of full bays can leave a remainder of a few millimetres and
    two members almost touching.
    """
    span = last - first
    if span <= 1e-6:
        return [(first + last) * 0.5]
    if span <= pitch:
        return [first, last]

    full = max(0, math.ceil((span - 2.0 * pitch) / pitch))
    edge = (span - full * pitch) * 0.5
    run = [first]
    run.extend(first + edge + i * pitch for i in range(full + 1))
    run.append(last)
    return run


def build_prism(profile, caps, low, high, point):
    """Extrude a 2D (along, z) profile between two positions across it.

    `caps` are index tuples tiling the profile into quads. They are supplied
    rather than derived because these profiles are non-convex - a rebated beam
    end, an L-shaped bracket - and the IFC tessellator fans from vertex 0,
    which triangulates a non-convex polygon wrongly.
    """
    count = len(profile)
    verts = [point(along, low, z) for along, z in profile]
    verts += [point(along, high, z) for along, z in profile]

    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j, count + i))
    for cap in caps:
        faces.append(tuple(reversed(cap)))
        faces.append(tuple(count + i for i in cap))
    return verts, faces


def build_cylinder_mesh(centre, radius, bottom_z, top_z, radial=16):
    verts = []
    for z in (bottom_z, top_z):
        for j in range(radial):
            theta = j / radial * math.tau
            verts.append(
                Vector(
                    (
                        centre[0] + math.cos(theta) * radius,
                        centre[1] + math.sin(theta) * radius,
                        z,
                    )
                )
            )
    faces = []
    for j in range(radial):
        j_next = (j + 1) % radial
        faces.append((j, j_next, radial + j_next, radial + j))
    faces.append(tuple(reversed(range(radial))))
    faces.append(tuple(range(radial, 2 * radial)))
    return verts, faces


# --------------------------------------------------------------------------
# cabin layout
# --------------------------------------------------------------------------


class WallLine:
    """One wall, described as a horizontal line the logs are laid along."""

    def __init__(self, name, axis, position, start, end, height):
        self.name = name
        self.axis = axis  # 'X' or 'Y'
        self.position = position
        self.start = start
        self.end = end
        self.height = height
        self.parity = 0 if axis == "X" else 1

    def endpoints(self, z):
        if self.axis == "X":
            return (
                Vector((self.start, self.position, z)),
                Vector((self.end, self.position, z)),
            )
        return (
            Vector((self.position, self.start, z)),
            Vector((self.position, self.end, z)),
        )

    def covers(self, coord):
        low, high = min(self.start, self.end), max(self.start, self.end)
        return low - 1e-6 <= coord <= high + 1e-6


def collect_wall_lines(props):
    over = props.overhang
    length = props.length
    width = props.width
    height = props.wall_height

    lines = [
        WallLine("Wall_South", "X", 0.0, -over, length + over, height),
        WallLine("Wall_North", "X", width, -over, length + over, height),
        WallLine("Wall_West", "Y", 0.0, -over, width + over, height),
        WallLine("Wall_East", "Y", length, -over, width + over, height),
    ]

    tie = props.internal_overhang
    for index, wall in enumerate(props.internal_walls):
        if wall.axis == "X":
            lines.append(
                WallLine(
                    f"Wall_Internal_{index + 1}",
                    "X",
                    wall.position,
                    -tie,
                    length + tie,
                    wall.height,
                )
            )
        else:
            lines.append(
                WallLine(
                    f"Wall_Internal_{index + 1}",
                    "Y",
                    wall.position,
                    -tie,
                    width + tie,
                    wall.height,
                )
            )
    return lines


def pile_positions(props):
    """Corner piles plus intermediates wherever a span exceeds max_span."""
    points = {}

    def run(x0, y0, x1, y1):
        span = math.dist((x0, y0), (x1, y1))
        segments = max(1, math.ceil(span / props.max_span))
        for i in range(segments + 1):
            t = i / segments
            key = (round(_lerp(x0, x1, t), 4), round(_lerp(y0, y1, t), 4))
            points[key] = True

    length = props.length
    width = props.width
    run(0.0, 0.0, length, 0.0)
    run(length, 0.0, length, width)
    run(length, width, 0.0, width)
    run(0.0, width, 0.0, 0.0)

    for wall in props.internal_walls:
        if wall.axis == "X":
            run(0.0, wall.position, length, wall.position)
        else:
            run(wall.position, 0.0, wall.position, width)

    return sorted(points)


def generate_cabin(props):
    """Build every log and pile. Returns (log_records, pile_positions).

    Every cut is worked out here, from the schedule alone. Nothing downstream
    needs to know about neighbouring geometry.
    """
    rng = random.Random(props.seed)

    r_butt = props.butt_diameter / 2.0
    r_top = props.top_diameter / 2.0
    pair = r_butt + r_top  # constant sum along the length, given alternation

    round_rise = pair - props.scribe_depth
    if round_rise <= 1e-4:
        raise ValueError(
            "Scribe depth is at or above the mean log diameter - logs would "
            "not stack. Reduce Scribe Depth."
        )
    half_rise = round_rise / 2.0

    lines = collect_wall_lines(props)

    # With sawn sills the first course sits with its axis *on* the foundation,
    # so its lower half is cut away and what remains is a half log bedded flat.
    # The second course, half a round higher, still dips below the foundation
    # and takes a shallower flat. That is what brings all four bottom logs down
    # onto the piles instead of leaving the second course hanging in the air.
    # Nothing above needs adjusting: only undersides are cut, and the courses
    # above scribe onto tops.
    base_z = 0.0 if props.sill_flat else r_butt
    max_courses = max(1, round(props.wall_height / half_rise))

    def course_z(course):
        return base_z + course * half_rise

    def lines_at(course):
        """(index, line) pairs carrying a log at this course."""
        if course < 0:
            return []
        z = course_z(course)
        return [
            (index, line)
            for index, line in enumerate(lines)
            if course % 2 == line.parity and z + pair / 2.0 <= line.height + 1e-6
        ]

    def flipped(course, wall_index):
        # Alternate butt and top ends so the wall rises level. Opposite walls
        # in the same course run opposite ways too.
        return props.alternate_butt and ((course // 2 + wall_index) % 2 == 1)

    def ends(line, course, wall_index):
        p_start, p_end = line.endpoints(course_z(course))
        if flipped(course, wall_index):
            p_start, p_end = p_end, p_start
        return p_start, p_end

    def radius_at(line, course, wall_index, coord):
        """Radius of that line's log where the given coordinate crosses it."""
        p_start, p_end = ends(line, course, wall_index)
        if line.axis == "X":
            span = p_end.x - p_start.x
            u = 0.0 if abs(span) < 1e-9 else (coord - p_start.x) / span
        else:
            span = p_end.y - p_start.y
            u = 0.0 if abs(span) < 1e-9 else (coord - p_start.y) / span
        return _lerp(r_butt, r_top, min(max(u, 0.0), 1.0))

    records = []
    by_course = {}  # course -> [(log, line, wall_index)], for tying the floor in
    for course in range(max_courses):
        below = lines_at(course - 1)
        two_below = {index for index, _line in lines_at(course - 2)}
        above = {index for index, _line in lines_at(course + 2)}

        # The real logs one and two courses down, so the groove and the
        # saddle notches can be cut to the surfaces that are actually there.
        # Two logs in a course share the same *nominal* size, so using the
        # schedule (radius_at, r_butt/r_top) here read as a rounding error
        # too small to matter - but jitter is drawn per log, so the crossing
        # log's real surface still differs from the schedule by a few
        # millimetres, and a crossing is exactly the one place two logs'
        # surfaces have to agree precisely. Every log in this course is
        # already built by the time we get here, since courses are processed
        # in order and this reads only courses already finished.
        below_logs = {
            index: member for member, _line, index in by_course.get(course - 1, [])
        }
        two_below_logs = {
            index: member for member, _line, index in by_course.get(course - 2, [])
        }

        for wall_index, line in lines_at(course):
            p_start, p_end = ends(line, course, wall_index)
            log = Log(
                p_start,
                p_end,
                r_butt,
                r_top,
                rng,
                props.bow,
                props.radial_jitter,
                props.bow_vertical,
                props.knot_density,
                props.knot_size,
                props.knot_rise,
            )

            # Lateral groove, cut to the log two courses down on the same wall.
            # Its taper runs opposite to this one whenever the ends alternate,
            # so a butt always beds onto a top.
            if wall_index in two_below:
                below_member = two_below_logs.get(wall_index)
                if below_member is not None:
                    # The real log's radius at each of this log's own ends,
                    # so a fat or thin neighbour is what the groove is cut to
                    # rather than what the schedule assumes is there.
                    coord0 = p_start.x if line.axis == "X" else p_start.y
                    coord1 = p_end.x if line.axis == "X" else p_end.y
                    log.groove = (
                        round_rise,
                        below_member.radius(
                            _param_along(below_member, line.axis, coord0)
                        ),
                        below_member.radius(
                            _param_along(below_member, line.axis, coord1)
                        ),
                    )
                else:
                    opposed = flipped(course - 2, wall_index) != flipped(
                        course, wall_index
                    )
                    if opposed:
                        log.groove = (round_rise, r_top, r_butt)
                    else:
                        log.groove = (round_rise, r_butt, r_top)

            # Saddle notches over every crossing log one course down.
            for cross_index, cross in below:
                if not cross.covers(line.position):
                    continue  # that wall does not reach this one
                if line.axis == "X":
                    offset = (cross.position - p_start.x) * log.dir.x
                else:
                    offset = (cross.position - p_start.y) * log.dir.y
                if offset < -r_butt or offset > log.length + r_butt:
                    continue  # crossing lies off the end of this log

                cross_member = below_logs.get(cross_index)
                if cross_member is None:
                    drop = half_rise
                    radius = radius_at(cross, course - 1, cross_index, line.position)
                else:
                    # The crossing log's own real radius and axis height at
                    # the point it is actually crossed, not the schedule's
                    # idea of where a nominal log's surface would be.
                    u = _param_along(cross_member, cross.axis, line.position)
                    drop = course_z(course) - cross_member.axis_point(u).z
                    radius = cross_member.radius(u)

                log.notches.append((offset, drop, radius, None))

            # The foundation plane is handed to every log as a floor. It only
            # bites on the two bottom courses, whose undersides fall below it;
            # higher up the log never reaches it and the cut never binds.
            if props.sill_flat:
                log.floor_z = 0.0

            # Top logs carry the roof, so they can be flattened to a level
            # bearing surface. Nothing sits on them to scribe against.
            if props.flatten_top and wall_index not in above:
                log.flat_z = course_z(course) + pair / 2.0 - props.top_flat_depth

            records.append((log, line.name, course))
            by_course.setdefault(course, []).append((log, line, wall_index))

    # ---- floor and ceiling ------------------------------------------------
    # The two are the same structure at different heights: members spanning
    # between opposite wall courses in whichever style is chosen. Only the
    # bearing course differs, and what sits around it - a floor is carried on
    # piles and locked down by the wall course above, a ceiling has neither.
    joists = []
    boxes = []
    floor_piles = []

    mean_r = pair / 2.0
    along_y = props.joist_axis == "Y"
    parity = 0 if along_y else 1  # the courses running across the joists

    levels = []
    if props.floor_type != "NONE":
        levels.append(("Floor", parity, True, True, None))

        # Bear the ceiling on the topmost course running the right way.
        top_course = None
        for course in range(max_courses - 1, -1, -1):
            if course % 2 == parity and lines_at(course):
                top_course = course
                break
        if props.add_ceiling and top_course is not None:
            if props.floor_type == "NOTCHED":
                # Drop the logs a round so the top wall course lands on top of
                # them and can be notched over them - the same lock-in the
                # floor logs get. It also buries the ceiling build-up within
                # the wall instead of stacking it above the head.
                ceiling_bearer = top_course - 2
            else:
                # Brackets hang off the wall head, so they bear on the top
                # course itself.
                ceiling_bearer = top_course

            if ceiling_bearer > parity:
                # Hung from a level worked out at the crown, so the deck over
                # the joists finishes level with it.
                sheet = props.sheet_thickness if props.add_deck else 0.0
                crown = course_z(ceiling_bearer) + mean_r
                levels.append(
                    (
                        "Ceiling",
                        ceiling_bearer,
                        False,
                        True,
                        crown - sheet - props.joist_depth,
                    )
                )

    for prefix, bearer_course, add_piles, tie_above, hung_base in levels:
        if along_y:
            span_end = props.width
            free_end = props.length
        else:
            span_end = props.length
            free_end = props.width

        # Every line at that course runs perpendicular to the joists, so this
        # picks up internal walls as well as the two external sills. A joist
        # crossing an internal wall gets notched over it just the same.
        bearers = lines_at(bearer_course)

        # The real sill logs, so the floor can be cut to the surfaces that are
        # actually there rather than to the nominal schedule. A beam is
        # thinner than what it beds on, so a neighbour's jitter and bow - up
        # to about a centimetre - would otherwise show as the beam sinking in.
        bearer_logs = {
            index: (member, line)
            for member, line, index in by_course.get(bearer_course, [])
        }
        walls = (0, 1) if along_y else (2, 3)

        sill_z = course_z(bearer_course)

        place = _frame_place(along_y)
        point = _frame_point(along_y)

        def add_box(name, lo, hi, kind):
            verts, faces = build_box_mesh(lo, hi)
            boxes.append((name, verts, faces, kind))

        def lay_sheets(label, frame, carriers, along_span, across_span, base, phase):
            """One layer of sheets over whatever members carry it.

            Each layer names its own carriers and direction, because they are
            not always the same: under a notched floor the lower sheet rides
            the logs, while the upper one rides the beams crossing them.
            """
            if not carriers:
                return
            put = _frame_place(frame)
            wide, long = props.sheet_width, props.sheet_length
            along_lo, along_hi = along_span
            across_lo, across_hi = across_span

            # Every seam running along the carriers lands on one, so both
            # sheets meeting there are supported. Take the furthest carrier
            # still within a sheet of the last seam rather than stepping a
            # fixed width. Where the carriers are further apart than a sheet -
            # the floor logs at 2 m - it falls back to a floating seam.
            bounds = [across_lo]
            for _ in range(len(carriers) + 2):
                if across_hi - bounds[-1] <= wide + 1e-6:
                    break
                nxt = None
                for centre in carriers:
                    if centre <= bounds[-1] + 1e-6:
                        continue
                    if centre - bounds[-1] > wide + 1e-6:
                        break
                    nxt = centre
                bounds.append(bounds[-1] + wide if nxt is None else nxt)
            bounds.append(across_hi)

            for row, (edge_a, edge_b) in enumerate(zip(bounds, bounds[1:])):
                if edge_b - edge_a <= 1e-6:
                    continue
                # Stagger alternate rows by half a sheet so the cross joints
                # break rather than running the length of the floor. The phase
                # also offsets one layer from the other.
                cursor = along_lo - ((row + phase) % 2) * long * 0.5
                column = 0
                while cursor < along_hi - 1e-6:
                    start = max(cursor, along_lo)
                    stop = min(cursor + long, along_hi)
                    cursor += long
                    if stop - start <= 1e-6:
                        continue
                    column += 1
                    add_box(
                        f"{label}_{row + 1:02d}_{column:02d}",
                        *put(
                            start, stop, edge_a, edge_b,
                            base, base + props.sheet_thickness,
                        ),
                        "sheet",
                    )

        if props.floor_type == "NOTCHED":
            # The outermost logs sit a set distance in from the walls, centre
            # to centre, and the run between them is on the nominal pitch with
            # the end bays absorbing the remainder - the bracket joists' rule
            # at a longer pitch, since these are primary members.
            positions = _even_run(
                props.log_beam_offset,
                free_end - props.log_beam_offset,
                props.log_beam_spacing,
            )
        else:
            # Bracket-hung joists are board-led. The outermost pair sits a
            # fixed gap off its wall; the run between them is on the nominal
            # pitch, with the two end bays absorbing whatever does not divide
            # evenly. Squeezing those two rather than spreading the error over
            # every bay keeps the boards on a true 60 cm grid across the room,
            # and no bay ends up wider than the pitch.
            inset = mean_r + props.joist_wall_gap + props.joist_width / 2.0
            positions = _even_run(
                inset, free_end - inset, props.joist_spacing
            )

        # Where the deck lands, what carries it, and which way those members
        # run. The bracket version mills the sills, which moves the wall face
        # along the beams inward.
        deck_layers = []
        face = mean_r

        if props.floor_type == "NOTCHED":
            level_beams = []  # this level's logs, for the tie-in below
            r_joist = props.joist_diameter / 2.0
            axis_z = sill_z + half_rise  # a course above its bearer, as a wall log would be

            # Only a quarter of the diameter comes off the top. Sawing down to
            # the middle would leave a thin plank of what ought to be a beam.
            joist_flat = (
                axis_z + r_joist - props.joist_diameter * props.joist_flat_share
            )

            # The wall course that comes to rest on the beams.
            above_course = bearer_course + 2
            above_axis_z = course_z(above_course)

            # Wall logs are cut to each other from the schedule, since every
            # log in a course is the same nominal size. A beam is not: it is
            # thinner than what it beds on, so the neighbour's jitter and bow
            # - up to about a centimetre - show up as the beam sinking into
            # the log it rests on. These are the real logs, so the beams can
            # be cut to the surfaces that are actually there.
            above_logs = {
                index: (member, line)
                for member, line, index in by_course.get(above_course, [])
            }

            for index, pos in enumerate(positions):
                # Run past the wall axes so each beam is let into the sill and
                # ends inside it, rather than stopping on the centreline.
                tie = props.joist_overhang
                start, end = place(
                    -tie, span_end + tie, pos, pos, axis_z, axis_z
                )
                log = Log(
                    start,
                    end,
                    r_joist,
                    r_joist,
                    rng,
                    props.bow,
                    props.radial_jitter,
                    props.bow_vertical,
                    props.knot_density,
                    props.knot_size,
                    props.knot_rise,
                )
                log.flat_z = joist_flat
                # The flat is full between the wall axes. Past them it runs
                # out along the underside of the log bearing on the beam,
                # instead of ending in a sawn cliff, and the projecting end
                # comes back to a full round like an overhanging corner log.
                log.flat_span = (tie, tie + span_end)
                shoulders = []
                for wall_index in walls:
                    entry = above_logs.get(wall_index)
                    if entry is None:
                        shoulders.append((mean_r, above_axis_z))
                        continue
                    member, line_above = entry
                    u = _param_along(member, line_above.axis, pos)
                    shoulders.append(
                        (member.radius(u), member.axis_point(u).z)
                    )
                log.flat_shoulders = tuple(shoulders)

                origin = start.y if along_y else start.x
                heading = log.dir.y if along_y else log.dir.x
                for bearer_index, bearer in bearers:
                    if not bearer.covers(pos):
                        continue  # that wall does not reach this joist
                    offset = (bearer.position - origin) * heading
                    if offset < -r_joist or offset > log.length + r_joist:
                        continue  # crossing lies off the end of this joist

                    entry = bearer_logs.get(bearer_index)
                    if entry is None:
                        drop, radius = half_rise, mean_r
                    else:
                        member, _line = entry
                        u = _param_along(member, bearer.axis, pos)
                        drop = axis_z - member.axis_point(u).z
                        radius = member.radius(u)
                    log.notches.append((offset, drop, radius, None))
                joists.append((log, f"{prefix}_Log_{index + 1:02d}", "joist"))
                level_beams.append(log)

                # Piles under the log, on the same grid as the perimeter, so
                # no unsupported span exceeds Max Pile Span. Its ends bear on
                # the sills, which carry their own piles already. A ceiling
                # has nothing to stand on, so it gets none.
                if add_piles:
                    segments = max(1, math.ceil(span_end / props.max_span))
                    for step in range(1, segments):
                        coord = span_end * step / segments
                        spot = (pos, coord) if along_y else (coord, pos)
                        floor_piles.append((round(spot[0], 4), round(spot[1], 4)))

            # The joists stand proud of the sills, so the wall course above has
            # to be notched over them - and that notch is what actually ties
            # the floor into the wall. The cut stops at the beam's sawn flat,
            # since there is no longer a full round up there to follow.
            joist_axis = "Y" if along_y else "X"
            tie_targets = by_course.get(above_course, []) if tie_above else []
            for log_above, line_above, index_above in tie_targets:
                if not 0.0 <= line_above.position <= span_end:
                    continue  # that wall does not pass over the joists
                start_above, _end_above = ends(
                    line_above, bearer_course + 2, index_above
                )
                origin_above = start_above.x if along_y else start_above.y
                heading_above = log_above.dir.x if along_y else log_above.dir.y
                for joist_index, pos in enumerate(positions):
                    if not line_above.covers(pos):
                        continue
                    beam = level_beams[joist_index]
                    # Both sides measured on the real members, so the seating
                    # matches what the beam was actually cut to.
                    u = _param_along(log_above, line_above.axis, pos)
                    v = _param_along(beam, joist_axis, line_above.position)
                    crown = log_above.axis_point(u).z
                    log_above.notches.append(
                        (
                            (pos - origin_above) * heading_above,
                            crown - beam.axis_point(v).z,
                            beam.radius(v),
                            joist_flat - crown,
                        )
                    )

            # The logs are too far apart for boards to span, so sawn beams run
            # across them, turned through ninety degrees. Same layering as the
            # bracket floor above this point, minus the brackets: these simply
            # bear on the logs' sawn flats.
            cross = _frame_place(not along_y)
            half_width = props.joist_width / 2.0
            cross_inset = mean_r + props.joist_wall_gap + half_width
            cross_at = _even_run(
                cross_inset, span_end - cross_inset, props.joist_spacing
            )
            cross_lo, cross_hi = mean_r, free_end - mean_r

            # A sheet goes down on the logs first, so the beams start one
            # thickness higher. The void it closes off between the beams is
            # what takes the insulation.
            sheet = props.sheet_thickness if props.add_deck else 0.0
            beam_base = joist_flat + sheet

            for index, seat in enumerate(cross_at):
                lo, hi = cross(
                    cross_lo, cross_hi,
                    seat - half_width, seat + half_width,
                    beam_base, beam_base + props.joist_depth,
                )
                add_box(f"{prefix}_Beam_{index + 1:02d}", lo, hi, "joist")

            # Both layers are set out to the beams, even though the lower one
            # rests on the logs. Its seams then run directly under a beam,
            # which pins both sheets meeting there - better than setting it
            # out to logs 2 m apart, where no seam could reach one at all.
            deck_run = (cross_lo, cross_hi)
            deck_bays = (mean_r, span_end - mean_r)
            deck_layers = [
                (f"{prefix}_SheetUnder", not along_y, cross_at, deck_run,
                 deck_bays, joist_flat, 1),
                (f"{prefix}_Sheet", not along_y, cross_at, deck_run,
                 deck_bays, beam_base + props.joist_depth, 0),
            ]

        else:  # BRACKET
            half_width = props.joist_width / 2.0
            plate = props.bracket_thickness
            # A floor stands up off the foundation: a sheet goes on underneath
            # too, so the structure rides one sheet thickness up and that lower
            # sheet finishes flush with the foundation top, which is also the
            # underside of the lowest wall course. A ceiling instead hangs from
            # a level worked out at the wall head.
            if hung_base is not None:
                underside = hung_base
            else:
                underside = props.sheet_thickness if props.add_deck else 0.0
            joist_top = underside + props.joist_depth

            # Mill the inner face of the sills the brackets bear against, down
            # their whole length, so there is a true surface to bolt to. Only
            # the bracket version gets this: a notched beam beds on the log's
            # own round surface and wants no flat. Doing it per bracket
            # instead changed the section abruptly at each pad's ends, which
            # tore the mesh; a continuous flat has no such transition, and
            # running the mill down the log once is what a builder would do.
            face = max(0.0, mean_r - props.bracket_mill)
            for wall_index in walls:
                entry = bearer_logs.get(wall_index)
                if entry is None:
                    continue
                member, line = entry
                across = (
                    Vector((0.0, 1.0, 0.0))
                    if line.axis == "X"
                    else Vector((1.0, 0.0, 0.0))
                )
                # Which side of the log faces indoors, in its own frame. Butt
                # alternation flips that frame, so it has to be measured.
                indoors = 1.0 if line.position <= 0.5 * span_end else -1.0
                limit = indoors * face * across.dot(member.side)
                reach = 1e3 if limit < 0.0 else -1e3
                member.side_flats.append(
                    (
                        member.length * 0.5,
                        member.length,
                        min(limit, reach),
                        max(limit, reach),
                    )
                )

            # Everything is measured out from the milled face. The beam is
            # rebated around the bracket rather than perched on it, so their
            # end faces and their undersides finish in the same planes - which
            # is what lets a sheet lie flat under the pair of them.
            bearing = props.joist_depth * 0.6
            seat_top = underside + plate
            plate_top = underside + props.joist_depth * 0.6

            near_end, near_plate = face, face + plate
            near_seat = near_plate + bearing
            far_end = span_end - face
            far_plate = far_end - plate
            far_seat = far_plate - bearing

            for index, pos in enumerate(positions):
                # A stepped prism: narrow through the seat band, wider past
                # the upright, full width clear of the bracket altogether.
                profile = [
                    (near_seat, underside),
                    (far_seat, underside),
                    (far_seat, seat_top),
                    (far_plate, seat_top),
                    (far_plate, plate_top),
                    (far_end, plate_top),
                    (far_end, joist_top),
                    (near_end, joist_top),
                    (near_end, plate_top),
                    (near_plate, plate_top),
                    (near_plate, seat_top),
                    (near_seat, seat_top),
                ]
                verts, faces = build_prism(
                    profile,
                    [(0, 1, 2, 11), (10, 3, 4, 9), (8, 5, 6, 7)],
                    pos - half_width,
                    pos + half_width,
                    point,
                )
                boxes.append(
                    (f"{prefix}_Joist_{index + 1:02d}", verts, faces, "joist")
                )

                # One L-shaped bracket per end, rather than a seat and an
                # upright meeting in mid air.
                ends = (
                    (near_end, near_plate, near_seat),
                    (far_end, far_plate, far_seat),
                )
                for side, (outer, inner, tail) in enumerate(ends):
                    profile = [
                        (outer, underside),
                        (tail, underside),
                        (tail, seat_top),
                        (inner, seat_top),
                        (inner, plate_top),
                        (outer, plate_top),
                    ]
                    verts, faces = build_prism(
                        profile,
                        [(0, 1, 2, 3), (0, 3, 4, 5)],
                        pos - half_width - plate,
                        pos + half_width + plate,
                        point,
                    )
                    boxes.append(
                        (
                            f"{prefix}_Bracket_{index + 1:02d}{'AB'[side]}",
                            verts,
                            faces,
                            "bracket",
                        )
                    )

            # Both layers ride the same joists here, so both keep the same
            # frame. Along the beams the wall face is the milled one.
            bracket_along = (face, span_end - face)
            bracket_across = (mean_r, free_end - mean_r)
            deck_layers = [
                (f"{prefix}_Sheet", along_y, positions, bracket_along,
                 bracket_across, joist_top, 0),
                (f"{prefix}_SheetUnder", along_y, positions, bracket_along,
                 bracket_across, underside - props.sheet_thickness, 1),
            ]

        # ---- plywood ---------------------------------------------------
        if props.add_deck:
            for layer in deck_layers:
                lay_sheets(*layer)

    # ---- roof ---------------------------------------------------------------
    # Gables carried up in logs on the wall's own course rhythm, so each one
    # keeps grooving onto the one below exactly as the wall does. A ridge and
    # purlins then span between the two gables.
    if props.add_roof:
        ridge_x = props.ridge_axis == "X"
        gable_parity = 1 if ridge_x else 0
        gable_lines = (2, 3) if ridge_x else (0, 1)
        slope_span = props.width if ridge_x else props.length
        run_end = props.length if ridge_x else props.width

        occupied = [c for c in range(max_courses) if lines_at(c)]
        gable_top = max(
            (c for c in occupied if c % 2 == gable_parity), default=None
        )

        if gable_top is not None and slope_span > 2.0 * pair:
            wall_top = course_z(max(occupied)) + mean_r
            pitch = math.radians(props.roof_pitch)
            half = slope_span * 0.5
            ridge_z = wall_top + half * math.tan(pitch)
            per_rise = 1.0 / math.tan(pitch)  # horizontal inset per unit rise

            # Run each log out past where the roof plane meets its axis, far
            # enough that the plane cuts clean through it. The end face is
            # then an ellipse lying in the roof plane, and the ends line up
            # into a rake. Each end runs out by its *own* radius: one figure
            # for both overshoots the thin end, pushing the rake below the
            # underside there, which collapses the section and tears the mesh.
            trim = 1.0
            out_butt = r_butt * per_rise * trim
            out_top = r_top * per_rise * trim
            rake = (math.tan(pitch), r_butt * trim, r_top * trim)

            def gable_span(course):
                """Where a gable log at this course starts and stops."""
                z = course_z(course)
                cut = (z - wall_top) * per_rise
                if cut <= 0.0 or half - cut < pair * 0.5:
                    return None
                return z, cut, slope_span - cut

            # The top wall log is what the first gable log grooves onto.
            below = {
                index: (-props.overhang, slope_span + props.overhang,
                        flipped(gable_top, index))
                for index in gable_lines
            }

            # Those same wall logs run on under the eaves to the building
            # corner, past where the roof plane - extended the same distance
            # beyond the wall as it is above - has already dropped below
            # their crown. Their outer tips poke out past the roof's own
            # silhouette unless they are raked the same way the triangle
            # above them is. How many courses that reaches depends on the
            # pitch and the overhang, not just the top one, so courses are
            # walked down from the top until one is found that the roof
            # plane clears entirely - every course under that one clears it
            # by even more, since a lower course only sits further back
            # under the roof.
            def roof_height(a):
                return wall_top + (half - abs(a - half)) * math.tan(pitch)

            deepest_course = None
            for course in range(gable_top, -1, -2):
                members = [
                    (member, member.p0.z - roof_height(
                        member.p0.y if ridge_x else member.p0.x
                    ), member.p0.z - roof_height(
                        member.p1.y if ridge_x else member.p1.x
                    ))
                    for member, _line, wall_index in by_course.get(course, [])
                    if wall_index in gable_lines
                ]
                if not members:
                    continue
                needed = any(
                    -drop_start < member.radius(0.0)
                    or -drop_end < member.radius(1.0)
                    for member, drop_start, drop_end in members
                )
                if not needed:
                    break
                deepest_course = course
                for member, drop_start, drop_end in members:
                    member.rake = (math.tan(pitch), drop_start, drop_end)

            # How far past the wall face the deck reaches on the way down:
            # out to where the rake cut would stop being needed one course
            # below the lowest one it actually touches, so the deck oversails
            # that log's own eave edge the way a verge or fascia would rather
            # than stopping flush on the last cut course.
            deck_overhang = 0.0
            if deepest_course is not None:
                crown_ref = course_z(max(0, deepest_course - 2)) + mean_r
                deck_overhang = min(
                    props.overhang, max(0.0, (wall_top - crown_ref) / math.tan(pitch))
                )

            gable_members = []  # (log, line index, low, high) for tying in
            course = gable_top + 2
            while True:
                reach = gable_span(course)
                if reach is None:
                    break
                z, low, high = reach

                for side, index in enumerate(gable_lines):
                    line = lines[index]
                    flip = flipped(course, index)
                    # p0 is always the butt, so the thicker end runs out
                    # further - whichever way round the log was laid.
                    if flip:
                        near, far = low - out_top, high + out_butt
                    else:
                        near, far = low - out_butt, high + out_top
                    if ridge_x:
                        p0 = Vector((line.position, near, z))
                        p1 = Vector((line.position, far, z))
                    else:
                        p0 = Vector((near, line.position, z))
                        p1 = Vector((far, line.position, z))
                    if flip:
                        p0, p1 = p1, p0

                    log = Log(
                        p0, p1, r_butt, r_top, rng,
                        props.bow, props.radial_jitter, props.bow_vertical,
                        props.knot_density, props.knot_size, props.knot_rise,
                    )
                    # The run-out above was worked out from the nominal
                    # radius, but radius() carries the thickness jitter too.
                    # A log a few per cent fat at one end is not cut through -
                    # it keeps a lens of end cap - and a thin one is cut back
                    # past its underside. Since the jitter differs per log,
                    # that leaves every end a different size and the rake line
                    # ragged. Stretch each end by what its own radius actually
                    # is, and take the drop from the same figure: the end then
                    # closes exactly, and it closes *in the roof plane*,
                    # because the extra run-out and the extra drop cancel.
                    stretch_start = (log.radius(0.0) - r_butt) * per_rise
                    stretch_end = (log.radius(1.0) - r_top) * per_rise
                    log.p0 = log.p0 - log.dir * stretch_start
                    log.p1 = log.p1 + log.dir * stretch_end
                    log.length = (log.p1 - log.p0).length
                    log.rake = (rake[0], log.radius(0.0), log.radius(1.0))

                    # Gable logs are all different lengths, so the taper
                    # identity the wall relies on does not hold here: the log
                    # below has to be measured at this one's own ends.
                    low_b, high_b, flip_b = below[index]
                    start, stop = (high_b, low_b) if flip_b else (low_b, high_b)
                    run = stop - start

                    def taper(coord, start=start, run=run):
                        share = 0.0 if abs(run) < 1e-9 else (coord - start) / run
                        return _lerp(r_butt, r_top, min(max(share, 0.0), 1.0))

                    log.groove = (
                        round_rise,
                        taper(p0.y if ridge_x else p0.x),
                        taper(p1.y if ridge_x else p1.x),
                    )
                    below[index] = (near, far, flip)
                    gable_members.append((log, index, near, far))
                    records.append((log, f"Wall_Gable_{'AB'[side]}", course))

                course += 2

            # Ridge, then purlins stepping down the rafter line from it. Each
            # sits with its crown on the roof plane, so rafters would bear on
            # all of them evenly.
            members = [(half, props.ridge_diameter, "Ridge")]
            step = 1
            while True:
                slide = step * props.purlin_spacing
                offset = slide * math.cos(pitch)
                if offset >= half - pair * 0.25:
                    break
                for direction, tag in ((-1.0, "A"), (1.0, "B")):
                    members.append(
                        (
                            half + direction * offset,
                            props.purlin_diameter,
                            f"Purlin_{step:02d}{tag}",
                        )
                    )
                step += 1

            gable_axis = "Y" if ridge_x else "X"

            def resting_course(across_pos, ceiling_z):
                """The topmost gable course, at or below ceiling_z, whose span
                reaches across_pos - the log a member here would actually bear
                on. None where nothing does, which happens near the peak once
                the gable courses have run out."""
                best = None
                for member, _index, low, high in gable_members:
                    if not low - 1e-6 <= across_pos <= high + 1e-6:
                        continue
                    where = _param_along(member, gable_axis, across_pos)
                    axis = member.axis_point(where).z
                    if axis <= ceiling_z and (best is None or axis > best[0]):
                        best = (axis, member.radius(where))
                return best

            for across, diameter, name in members:
                radius = diameter * 0.5
                roof_z = ridge_z - abs(across - half) * math.tan(pitch) - radius
                z = roof_z

                # Nest into the top half of whichever course this member is
                # actually resting on, rather than trusting the continuous
                # roof-plane formula to land it there by itself. Gable courses
                # step by half_rise while the roof plane is continuous, so
                # left alone a member can end up low against a course - close
                # to daylight at the step above it - or, near the peak where
                # the last course is a sliver, floating clear of any gable
                # material at all. Only ever raised, never lowered: the crown
                # stays on or above the roof plane, never below it.
                rest = resting_course(across, z + radius)
                if rest is not None:
                    rest_axis, rest_radius = rest
                    z = max(z, rest_axis + rest_radius * 0.45)

                if ridge_x:
                    p0 = Vector((-props.overhang, across, z))
                    p1 = Vector((run_end + props.overhang, across, z))
                else:
                    p0 = Vector((across, -props.overhang, z))
                    p1 = Vector((across, run_end + props.overhang, z))

                beam = Log(
                    p0, p1, radius, radius, rng,
                    props.bow, props.radial_jitter, props.bow_vertical,
                    props.knot_density, props.knot_size, props.knot_rise,
                )

                # The crown already sits tangent to the roof plane, but a
                # circle's own tangent at its top is flat and the roof plane
                # is not - so the plane cuts through the barrel rather than
                # grazing past it, on whichever side runs downhill. slope is
                # d(roof height)/d(a): the roof plane's own downhill slope
                # (-tan(pitch) on the near side of the ridge, the sign flip
                # at the ridge itself is why the ridge beam needs both),
                # converted from world Y/X into this log's own `a` via the
                # matching component of its `side` vector.
                # The intercept anchors the cut to the *true* roof plane
                # (roof_z, before the top-half snap above), not to wherever
                # this beam's own crown now sits - the snap raises the axis
                # for seating, and the cut has to stay flush with the roof
                # regardless of how far that raised it.
                side_component = beam.side.y if ridge_x else beam.side.x
                downhill = -math.tan(pitch) * side_component
                # b at a=0 such that z + b lands exactly on the true roof
                # surface. roof_z is the *axis* height the unsnapped formula
                # gives (it already has radius subtracted), so the surface
                # itself is roof_z + radius.
                crown_b = roof_z + radius - z
                if abs(across - half) < 1e-9:
                    beam.tilt = [(downhill, crown_b), (-downhill, crown_b)]
                else:
                    sign = 1.0 if across > half else -1.0
                    beam.tilt = [(downhill * sign, crown_b)]

                # The gable is worked to receive the purlin, never the other
                # way round - it is the wall that gets cut, as the reference
                # detail has it. Which cut depends on where the log sits:
                #
                #   below the purlin - it is let in from above, so a seat;
                #   above it         - it rides over, so a notch from below.
                #
                # Seating both would strip the logs above the purlin away
                # entirely and leave daylight over it; notching both would cut
                # the wrong side of a wedge the rake has already thinned.
                for member, index, low, high in gable_members:
                    if not low - 1e-6 <= across <= high + 1e-6:
                        continue
                    where = _param_along(member, gable_axis, across)
                    axis = member.axis_point(where).z
                    reach = member.radius(where)
                    if axis + reach <= z - radius or axis - reach >= z + radius:
                        continue  # passes clear, above or below

                    if axis < z:
                        member.pockets.append(
                            (
                                where * member.length,
                                radius + props.scribe_gap,
                                z,
                            )
                        )
                    else:
                        # Near its own end this log has already been cut away
                        # from above by the rake. If that has taken it below
                        # the true roof plane at this point it never reaches
                        # the purlin, and notching it from below as well would
                        # consume what little of the wedge is left. Checked
                        # against the roof plane itself, not against z: z can
                        # sit above the roof plane once nested into a course
                        # below, and comparing against that inflated height
                        # would call a course clear that still has material
                        # over the purlin's actual, unnested position.
                        span_here = where * member.length
                        if member.rake is not None:
                            fall, drop_start, drop_end = member.rake
                            rake_h = min(
                                member.p0.z - drop_start + span_here * fall,
                                member.p0.z
                                - drop_end
                                + (member.length - span_here) * fall,
                            )
                            if rake_h <= roof_z + radius:
                                continue

                        member.notches.append(
                            (span_here, axis - z, radius, None)
                        )

                joists.append((beam, f"Roof_{name}", "purlin"))

            # Shared frame for anything laid on the roof plane itself - the
            # deck and the rafters both work in (along the ridge, down the
            # slope, side) rather than world XYZ.
            z_axis = Vector((0.0, 0.0, 1.0))
            along_axis = Vector((1.0, 0.0, 0.0)) if ridge_x else Vector((0.0, 1.0, 0.0))
            across_axis = Vector((0.0, 1.0, 0.0)) if ridge_x else Vector((1.0, 0.0, 0.0))
            cos_p, sin_p = math.cos(pitch), math.sin(pitch)
            # Slope distance from the ridge down to where the wall's own
            # rake cut stops being needed, past the wall face by however far
            # into the overhang that boundary actually sits.
            d_max = (half + deck_overhang) / cos_p
            along_lo, along_hi = -props.overhang, run_end + props.overhang

            def roof_point(along, d, s):
                return (
                    along_axis * along
                    + across_axis * (half + s * d * cos_p)
                    + z_axis * (ridge_z - d * sin_p)
                )

            def roof_normal(s):
                return across_axis * (s * sin_p) + z_axis * cos_p

            # Plywood sheathing, one slope at a time. Sheets tile the same
            # way the floor's do - along the beams on the nominal pitch, rows
            # across them snapped onto whichever carrier (ridge or purlin)
            # sits closest, so seams land on something rather than floating -
            # just worked out on the roof plane's own tilted frame instead of
            # a level one.
            if props.add_roof_deck:
                # The ridge log's own crown sits exactly on the true peak,
                # where both roof planes meet, but its flat cut face - what a
                # sheet actually beds onto - runs on from there out to its own
                # side, a run of radius*sin(pitch) measured down the slope.
                # A seam landing at the peak itself would rest on nothing;
                # landing at the centre of that face gives it real bearing on
                # both sides, the same reason a seam snaps onto a purlin.
                d_ridge_cut = (props.ridge_diameter * 0.5) * sin_p

                def lay_roof_sheets(label, s, carriers, lap, lift=0.0):
                    normal = roof_normal(s)
                    wide, long = props.sheet_width, props.sheet_length

                    # A sheet lifted clear of the roof plane no longer meets
                    # its mirror at d=0 - each side has to reach out over the
                    # peak by lift*tan(pitch) first, the same relation the
                    # rafters' own miter uses, before the two actually touch.
                    # At lift=0 (the deck itself) that reach is zero and this
                    # is exactly the old, already-proven ridge line.
                    d_min = -lift * math.tan(pitch)

                    # The first row is the thin strip from the true peak out
                    # to the centre of the ridge's own cut face - too narrow
                    # to be a sheet in its own right, but real coverage, not
                    # a seam. Every row after that steps to the next carrier
                    # inside reach of a sheet width rather than a fixed step,
                    # the same rule the floor uses, so a seam lands on a
                    # purlin wherever one is close enough, and falls back to
                    # a floating seam only where none is.
                    #
                    # `lap` carries this side's own strip past where the two
                    # actually meet and over the other slope's, rather than
                    # butting right on that line - the plane it stays in is
                    # still this side's own, so past the peak it rises clear
                    # of the other slope's surface instead of cutting into
                    # it, exactly the way a real board would sit proud over
                    # what it laps.
                    bounds = [d_min - lap, d_ridge_cut]
                    for _ in range(len(carriers) + 2):
                        if d_max - bounds[-1] <= wide + 1e-6:
                            break
                        nxt = None
                        for centre in carriers:
                            if centre <= bounds[-1] + 1e-6:
                                continue
                            if centre - bounds[-1] > wide + 1e-6:
                                break
                            nxt = centre
                        bounds.append(bounds[-1] + wide if nxt is None else nxt)
                    bounds.append(d_max)

                    for row, (d_a, d_b) in enumerate(zip(bounds, bounds[1:])):
                        if d_b - d_a <= 1e-6:
                            continue
                        cursor = along_lo - (row % 2) * long * 0.5
                        column = 0
                        while cursor < along_hi - 1e-6:
                            start = max(cursor, along_lo)
                            stop = min(cursor + long, along_hi)
                            cursor += long
                            if stop - start <= 1e-6:
                                continue
                            column += 1
                            corners = [
                                roof_point(start, d_a, s) + normal * lift,
                                roof_point(stop, d_a, s) + normal * lift,
                                roof_point(stop, d_b, s) + normal * lift,
                                roof_point(start, d_b, s) + normal * lift,
                            ]
                            verts, faces = build_slab_mesh(
                                corners, normal, props.sheet_thickness
                            )
                            boxes.append(
                                (
                                    f"{label}_{row + 1:02d}_{column:02d}",
                                    verts,
                                    faces,
                                    "sheet",
                                )
                            )

                for s, tag in ((-1.0, "A"), (1.0, "B")):
                    carriers = sorted(
                        {
                            abs(across - half) / cos_p
                            for across, _diameter, _name in members
                            if (across - half) * s >= -1e-9
                        }
                    )
                    lap = props.sheet_thickness if tag == "B" else 0.0
                    lay_roof_sheets(f"RoofDeck_{tag}", s, carriers, lap)

            # Rafters on top of the deck, running down the slope. Spaced the
            # same way the floor's joists are - the outermost pair flush
            # with the gable overhang, the rest on the nominal pitch, the
            # two end bays absorbing whatever is left over. Each position is
            # really two pieces, one per roof side, meeting near the ridge
            # the same way the deck's own sheets do - laid up from whichever
            # side the deck's own sheets are not, so the two layers break
            # joint rather than doubling up on the same side.
            if props.add_rafters:
                r_rafter = props.rafter_diameter * 0.5

                def rafter_rise(s):
                    return roof_normal(s) * (
                        props.sheet_thickness + props.rafter_height * 0.5
                    )

                # "Flush with the overhang" means faces flush, not centres:
                # the outermost rafter's own centre has to sit a radius in
                # from along_lo/along_hi, or its round side runs on past
                # where the purlins, ridge and deck - all referenced by
                # their own tips there - actually end.
                positions = _even_run(
                    along_lo + r_rafter, along_hi - r_rafter, props.rafter_spacing
                )
                rafters = {}  # (index, tag) -> Log, for the noggins below
                for index, along in enumerate(positions):
                    for s, tag in ((-1.0, "A"), (1.0, "B")):
                        # rise carries the axis sideways as well as up - its
                        # own normal is not vertical either - so reaching it
                        # from d=0 lands short of the true ridge line by
                        # exactly that sideways carry. Starting the same
                        # distance further out first cancels it, landing the
                        # axis pinned to the ridge line the miter is measured
                        # from. But a plain miter meets face to face, not
                        # edge to edge: it is the plank's own top, a full
                        # half-height above the axis, that has to reach the
                        # ridge line, not the axis itself - short of that,
                        # the top stays flat and square a while longer,
                        # showing as an uncut lip above the angled cut.
                        # log.up is the roof's own normal, so a further
                        # sideways carry of exactly the same shape - half the
                        # milled height, scaled by the same tan(pitch) - is
                        # what the top edge needs on top of the axis's own.
                        rise = rafter_rise(s)
                        rise_mag = props.sheet_thickness + props.rafter_height * 0.5
                        d_start = -math.tan(pitch) * (rise_mag + props.rafter_height * 0.5)
                        p0 = roof_point(along, d_start, s) + rise
                        p1 = roof_point(along, d_max, s) + rise
                        rafter = Log(
                            p0, p1, r_rafter, r_rafter, rng,
                            props.bow, props.radial_jitter, props.bow_vertical,
                            props.knot_density, props.knot_size, props.knot_rise,
                        )
                        rafter.plank = props.rafter_height
                        # Both sides cut to the same plane - the ridge line
                        # itself - so they close into one flush seam rather
                        # than one lapping over the other the way the deck's
                        # own sheets do.
                        rafter.miter = (across_axis, half)
                        rafters[index, tag] = rafter
                        joists.append(
                            (rafter, f"Roof_Rafter_{index + 1:02d}{tag}", "rafter")
                        )

                # Noggins between adjacent rafters at the bottom - sawn
                # boards, not logs, standing on edge across the full rafter
                # height, at 90 degrees to the deck the same way a joist
                # stands on edge across a floor. Each end is cut into the
                # rafter's own round side with the same push-to-surface
                # solve _cope_cap uses for a purlin's crown, just aimed
                # sideways at a neighbour instead of up at the roof plane -
                # only four corners of it rather than a whole ring, since a
                # board's end is flat to begin with.
                def noggin_point(along, thick, height, s):
                    # thick=1 (the downhill face) sits flush with d_max,
                    # where the rafters themselves end, rather than
                    # straddling it and overhanging past their tails.
                    d = d_max - (1 - thick) * props.noggin_thickness
                    return roof_point(along, d, s) + roof_normal(s) * (
                        props.sheet_thickness + height * props.rafter_height
                    )

                for index in range(len(positions) - 1):
                    a0, a1 = positions[index], positions[index + 1]
                    for s, tag in ((-1.0, "A"), (1.0, "B")):
                        left = rafters[index, tag]
                        right = rafters[index + 1, tag]
                        cope0 = (left.p0, left.dir, left.r0)
                        cope1 = (right.p0, right.dir, right.r0)
                        verts, faces = build_board_mesh(
                            lambda along, thick, height, s=s: noggin_point(
                                along, thick, height, s
                            ),
                            along_axis,
                            a0,
                            a1,
                            cope0,
                            cope1,
                            # a0/a1 are the rafters' own centres, not a safe
                            # distance short of them, so each end is pulled
                            # in from there rather than reached out to.
                            outward0=True,
                            outward1=False,
                            height_segments=8,
                        )
                        boxes.append(
                            (
                                f"Roof_Noggin_{index + 1:02d}{tag}",
                                verts,
                                faces,
                                "rafter",
                            )
                        )

                # Everything above the rafters stacks directly on whatever
                # is actually enabled below it - a second sheathing layer,
                # then laths opening a ventilated gap, then whichever
                # covering is selected - the same running-offset idea the
                # rafters' own rise already uses to clear the deck below.
                lift = props.sheet_thickness + props.rafter_height

                if props.add_roof_sheathing2:
                    for s, tag in ((-1.0, "A"), (1.0, "B")):
                        carriers = sorted(
                            {
                                abs(across - half) / cos_p
                                for across, _diameter, _name in members
                                if (across - half) * s >= -1e-9
                            }
                        )
                        lap = props.sheet_thickness if tag == "B" else 0.0
                        lay_roof_sheets(
                            f"RoofSheathing2_{tag}", s, carriers, lap, lift=lift
                        )
                    lift += props.sheet_thickness

                if props.add_roof_laths:
                    # Lifted clear of the roof plane, laths don't meet their
                    # mirror at d=0 either - same reach as the sheets above.
                    d_min = -lift * math.tan(pitch)
                    half_w = props.lath_width * 0.5
                    for index, along in enumerate(positions):
                        for s, tag in ((-1.0, "A"), (1.0, "B")):
                            normal = roof_normal(s)
                            corners = [
                                roof_point(along - half_w, d_min, s) + normal * lift,
                                roof_point(along + half_w, d_min, s) + normal * lift,
                                roof_point(along + half_w, d_max, s) + normal * lift,
                                roof_point(along - half_w, d_max, s) + normal * lift,
                            ]
                            verts, faces = build_slab_mesh(
                                corners, normal, props.lath_thickness
                            )
                            boxes.append(
                                (
                                    f"Roof_Lath_{index + 1:02d}{tag}",
                                    verts,
                                    faces,
                                    "lath",
                                )
                            )
                    lift += props.lath_thickness

                def lay_ridge_cap(c0, tilt_c, wood):
                    """A cap over the ridge: two wings, one per slope, whose
                    underside lies on the plane offset c0 from the roof at
                    the ridge line and climbs tilt_c per unit of slope, so it
                    follows the covering it sits on. Both are cut off along
                    the vertical plane through the ridge line, meeting in a
                    mitre. Wood is made of boards, each two half-thickness
                    layers so the joint between boards is a half-lap and
                    stays watertight; metal is one sheet per slope. Slots
                    through the wings are the ventilation."""
                    width = props.ridge_cap_width
                    reach = cos_p + tilt_c * sin_p
                    total = (
                        props.ridge_cap_wood_thickness
                        if wood
                        else props.ridge_cap_metal_thickness
                    )
                    lap = min(props.ridge_cap_lap, props.ridge_cap_piece_length * 0.5)

                    edges = [along_lo]
                    if wood:
                        x = along_lo + props.ridge_cap_piece_length
                        while x < along_hi - 0.5:
                            edges.append(x)
                            x += props.ridge_cap_piece_length
                    edges.append(along_hi)

                    slots = []
                    if props.add_ridge_vents:
                        vent_len = props.ridge_vent_length
                        pitch_v = max(props.ridge_vent_pitch, vent_len * 1.5)
                        x = along_lo + pitch_v * 0.5
                        while x + vent_len * 0.5 < along_hi - 0.05:
                            lo_v, hi_v = x - vent_len * 0.5, x + vent_len * 0.5
                            x += pitch_v
                            # Never through a joint, where boards overlap.
                            if any(
                                lo_v < joint + lap + 0.03 and hi_v > joint - 0.03
                                for joint in edges[1:-1]
                            ):
                                continue
                            slots.append((lo_v, hi_v))
                    vent_w = min(props.ridge_vent_width, width * 0.6)
                    slot_d = (width * 0.5 - vent_w * 0.5, width * 0.5 + vent_w * 0.5)

                    def layer(s, a0, a1, base, thickness):
                        normal = roof_normal(s)
                        d_bot = -base * sin_p / reach
                        d_top = -(base + thickness) * sin_p / reach
                        inside = [sl for sl in slots if a0 < sl[0] and sl[1] < a1]
                        rows = sorted({a0, a1, *[v for sl in inside for v in sl]})
                        if inside:
                            cols_bot = [d_bot, slot_d[0], slot_d[1], width]
                            cols_top = [d_top, slot_d[0], slot_d[1], width]
                        else:
                            cols_bot, cols_top = [d_bot, width], [d_top, width]
                        bot = [
                            [
                                roof_point(a, d, s) + normal * (base + tilt_c * d)
                                for d in cols_bot
                            ]
                            for a in rows
                        ]
                        top = [
                            [
                                roof_point(a, d, s)
                                + normal * (base + thickness + tilt_c * d)
                                for d in cols_top
                            ]
                            for a in rows
                        ]
                        skip = [
                            (i, 1)
                            for i in range(len(rows) - 1)
                            if any(
                                sl[0] - 1e-9 <= rows[i] and rows[i + 1] <= sl[1] + 1e-9
                                for sl in inside
                            )
                        ]
                        return build_grid_slab(bot, top, skip, flip=s < 0)

                    for s, tag in ((-1.0, "A"), (1.0, "B")):
                        if not wood:
                            verts, faces = layer(s, along_lo, along_hi, c0, total)
                            boxes.append(
                                (f"Roof_RidgeCap_Metal_{tag}", verts, faces, "covering")
                            )
                            continue
                        half_t = total * 0.5
                        last = len(edges) - 2
                        for k in range(len(edges) - 1):
                            upper = layer(s, edges[k], edges[k + 1], c0 + half_t, half_t)
                            lower = layer(
                                s,
                                edges[k] + (lap if k > 0 else 0.0),
                                edges[k + 1] + (lap if k < last else 0.0),
                                c0,
                                half_t,
                            )
                            verts, faces = merge_meshes([lower, upper])
                            boxes.append(
                                (
                                    f"Roof_RidgeCap_Wood_{tag}_{k + 1:02d}",
                                    verts,
                                    faces,
                                    "covering",
                                )
                            )

                if props.roof_covering in ("SHINGLES", "STONE"):
                    prefix = (
                        "Roof_Shingle"
                        if props.roof_covering == "SHINGLES"
                        else "Roof_Stone"
                    )
                    tile_w = props.covering_tile_width
                    tile_len = props.covering_tile_length
                    exposure = min(props.covering_exposure, tile_len)
                    thick = props.covering_tile_thickness

                    # Counter-battens - what a shingle or a stone tile is
                    # actually nailed to - sit at an even gauge from the
                    # ridge out to the eave: the exposure, nudged so a
                    # whole number of courses divides the slope exactly,
                    # the way a slater sets out. Every course's head lands
                    # on one of them, so every course has the same tilt.
                    gaps = max(1, round(d_max / exposure))
                    gauge = d_max / gaps
                    battens = [i * gauge for i in range(gaps + 1)]
                    tile_lift = lift + (
                        props.counter_batten_thickness
                        if props.add_counter_battens
                        else 0.0
                    )
                    if props.add_counter_battens:
                        half_bw = props.counter_batten_width * 0.5
                        for s, tag in ((-1.0, "A"), (1.0, "B")):
                            normal = roof_normal(s)
                            for row, d_raw in enumerate(battens):
                                # The ridge course's own head reaches back
                                # past d=0 on a plane that only clears the
                                # batten from d=0 on, so that one sits just
                                # short of the ridge rather than straddling it.
                                d_pos = max(d_raw, half_bw)
                                top = None
                                if row == gaps:
                                    # The eave batten sits flush with the
                                    # sheeting's edge, not centred on it, and
                                    # is raised to carry the last courses'
                                    # tails: its top follows the tilt of the
                                    # tile underside resting on it, so the
                                    # two meet with no gap.
                                    d_pos = d_raw - half_bw
                                    tail_tilt = thick / gauge
                                    d_last = battens[gaps - 1]

                                    def rest(d):
                                        return (
                                            tile_lift + tail_tilt * (d - d_last)
                                        )

                                    top = [
                                        roof_point(along_lo, d_pos - half_bw, s)
                                        + normal * rest(d_pos - half_bw),
                                        roof_point(along_hi, d_pos - half_bw, s)
                                        + normal * rest(d_pos - half_bw),
                                        roof_point(along_hi, d_pos + half_bw, s)
                                        + normal * rest(d_pos + half_bw),
                                        roof_point(along_lo, d_pos + half_bw, s)
                                        + normal * rest(d_pos + half_bw),
                                    ]
                                corners = [
                                    roof_point(along_lo, d_pos - half_bw, s)
                                    + normal * lift,
                                    roof_point(along_hi, d_pos - half_bw, s)
                                    + normal * lift,
                                    roof_point(along_hi, d_pos + half_bw, s)
                                    + normal * lift,
                                    roof_point(along_lo, d_pos + half_bw, s)
                                    + normal * lift,
                                ]
                                verts, faces = build_slab_mesh(
                                    corners,
                                    normal,
                                    props.counter_batten_thickness,
                                    top,
                                )
                                boxes.append(
                                    (
                                        f"Roof_CounterBatten_{tag}_{row + 1:02d}",
                                        verts,
                                        faces,
                                        "lath",
                                    )
                                )

                    # Every course lies on the same tilted plane relative to
                    # its own head batten: bedded on the batten at its head,
                    # and one thickness up by the next batten down, where
                    # it rides on the head of the course below. That fixes
                    # the tilt at thickness / gauge - the plane each course
                    # lies in, so neighbours meet face to face with no gap
                    # and no overlap - and it is the same for every course,
                    # the shortened eave ones included, so nothing drifts
                    # off the roof plane and nothing kinks at either end.
                    tilt = thick / gauge

                    def tile_offset(d, k):
                        return tile_lift + tilt * (d - battens[k])

                    # At the ridge the top course reaches back over the peak
                    # on the same plane, and both slopes are cut off along
                    # the vertical plane through the ridge line itself - a
                    # mitre - so the two meet flush in a point instead of
                    # one poking past the other. Solving d*cos + offset*sin
                    # = 0 for the plane's bottom and top faces gives where
                    # each of them reaches it.
                    reach = cos_p + tilt * sin_p
                    d_miter_lo = -tile_lift * sin_p / reach
                    d_miter_hi = -(tile_lift + thick) * sin_p / reach

                    for s, tag in ((-1.0, "A"), (1.0, "B")):
                        normal = roof_normal(s)
                        for row, d_head in enumerate(battens[:-1]):
                            d_lo, d_top = (
                                (d_miter_lo, d_miter_hi) if row == 0 else (d_head, d_head)
                            )
                            d_hi = min(d_max, d_head + tile_len)
                            cursor = along_lo - (row % 2) * tile_w * 0.5
                            column = 0
                            while cursor < along_hi - 1e-6:
                                start = max(cursor, along_lo)
                                stop = min(cursor + tile_w, along_hi)
                                cursor += tile_w
                                if stop - start <= 1e-6:
                                    continue
                                column += 1
                                off_lo = tile_offset(d_lo, row)
                                off_top = tile_offset(d_top, row) + thick
                                off_hi = tile_offset(d_hi, row)
                                corners = [
                                    roof_point(start, d_lo, s) + normal * off_lo,
                                    roof_point(stop, d_lo, s) + normal * off_lo,
                                    roof_point(stop, d_hi, s) + normal * off_hi,
                                    roof_point(start, d_hi, s) + normal * off_hi,
                                ]
                                top = [
                                    roof_point(start, d_top, s) + normal * off_top,
                                    roof_point(stop, d_top, s) + normal * off_top,
                                    roof_point(stop, d_hi, s)
                                    + normal * (off_hi + thick),
                                    roof_point(start, d_hi, s)
                                    + normal * (off_hi + thick),
                                ]
                                verts, faces = build_slab_mesh(
                                    corners, normal, thick, top
                                )
                                boxes.append(
                                    (
                                        f"{prefix}_{tag}_{row + 1:02d}_{column:02d}",
                                        verts,
                                        faces,
                                        "covering",
                                    )
                                )

                    if props.add_ridge_cap:
                        # Bedded a hair above the top course's own outer
                        # face, on a plane parallel to it.
                        lay_ridge_cap(
                            tile_lift + thick + 0.004,
                            tilt,
                            props.roof_covering == "SHINGLES",
                        )

                elif props.roof_covering == "METAL":
                    thick = props.covering_metal_thickness
                    depth = props.covering_corrugation_depth
                    # The wave's own peak, not just the panel's base, is what
                    # has to clear the far slope's - reaching from that
                    # keeps the crest gap-free at the cost of a sliver left
                    # at the troughs, the one place this profile can't meet
                    # its mirror and stay a plain sheared cut both sides.
                    d_min_metal = -(lift + thick + depth) * math.tan(pitch)
                    for s, tag in ((-1.0, "A"), (1.0, "B")):
                        normal = roof_normal(s)
                        verts, faces = build_corrugated_slab(
                            along_lo,
                            along_hi,
                            d_min_metal,
                            d_max,
                            lambda a, d, s=s: roof_point(a, d, s),
                            normal,
                            lift,
                            thick,
                            props.covering_corrugation_pitch,
                            depth,
                        )
                        boxes.append(
                            (f"Roof_Covering_Metal_{tag}", verts, faces, "covering")
                        )

                    if props.add_ridge_cap:
                        # Over the crests, which stand the corrugation depth
                        # proud of the panel's own plane.
                        lay_ridge_cap(lift + depth + 0.004, 0.0, False)

    piles = []
    if props.pile_height > 0.0:
        points = dict.fromkeys(pile_positions(props), True)
        points.update(dict.fromkeys(floor_piles, True))
        piles = sorted(points)

    return records, piles, joists, boxes


# --------------------------------------------------------------------------
# Blender object creation
# --------------------------------------------------------------------------


def _clear_collection():
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is None:
        return None
    for obj in list(coll.objects):
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return coll


def _get_collection(context):
    coll = _clear_collection()
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION_NAME)
        context.scene.collection.children.link(coll)
    return coll


def _apply_shading(obj, smooth, creases=()):
    """Smooth shading that keeps real creases sharp.

    `creases` are vertex index pairs known to be creases whatever their angle.
    """
    mesh = obj.data
    if not smooth:
        mesh.polygons.foreach_set("use_smooth", [False] * len(mesh.polygons))
        mesh.update()
        return

    bm = bmesh.new()
    bm.from_mesh(mesh)
    # The barrel and the scribed groove are curved. The rim where a notch
    # breaks the surface, and the sawn ends, are not.
    crease = math.radians(38.0)
    for face in bm.faces:
        face.smooth = True
    for edge in bm.edges:
        edge.smooth = edge.calc_face_angle(math.pi) < crease
    if creases:
        bm.verts.ensure_lookup_table()
        for i, j in creases:
            edge = bm.edges.get((bm.verts[i], bm.verts[j]))
            if edge is not None:
                edge.smooth = False
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _make_object(coll, name, verts, faces, obj_type, wall_name="", course=-1):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [list(f) for f in faces])
    mesh.validate(verbose=False)

    # Outward normals, which IFC tessellation wants. recalc_face_normals only
    # guarantees *consistency*, so the signed volume is checked as well:
    # negative means the shell came out inside in.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bm.calc_volume(signed=True) < 0.0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    obj[TYPE_KEY] = obj_type
    if wall_name:
        obj[WALL_KEY] = wall_name
    if course >= 0:
        obj[COURSE_KEY] = course
    coll.objects.link(obj)
    return obj


# --------------------------------------------------------------------------
# properties
# --------------------------------------------------------------------------


class LOGCABIN_InternalWall(bpy.types.PropertyGroup):
    axis: EnumProperty(
        name="Axis",
        items=[
            ("X", "Along X", "Runs parallel to the length, ties into the east and west walls"),
            ("Y", "Along Y", "Runs parallel to the width, ties into the north and south walls"),
        ],
        default="X",
    )
    position: FloatProperty(
        name="Position",
        description="Offset from the origin, measured across the wall's axis",
        default=2.5,
        min=0.0,
        unit="LENGTH",
    )
    height: FloatProperty(name="Height", default=2.4, min=0.1, unit="LENGTH")


class LOGCABIN_Props(bpy.types.PropertyGroup):
    # footprint
    length: FloatProperty(name="Length (X)", default=8.0, min=1.0, unit="LENGTH")
    width: FloatProperty(name="Width (Y)", default=5.0, min=1.0, unit="LENGTH")
    wall_height: FloatProperty(name="Wall Height", default=2.6, min=0.3, unit="LENGTH")
    overhang: FloatProperty(
        name="Corner Overhang",
        description="How far log ends project past the corner",
        default=0.35,
        min=0.0,
        unit="LENGTH",
    )

    # foundation
    max_span: FloatProperty(
        name="Max Pile Span",
        description="Insert extra piles whenever an unsupported span exceeds this",
        default=2.5,
        min=0.5,
        unit="LENGTH",
    )
    pile_diameter: FloatProperty(name="Pile Diameter", default=0.3, min=0.05, unit="LENGTH")
    pile_height: FloatProperty(name="Pile Height", default=0.6, min=0.0, unit="LENGTH")

    # logs
    butt_diameter: FloatProperty(name="Butt Diameter", default=0.30, min=0.05, unit="LENGTH")
    top_diameter: FloatProperty(name="Top Diameter", default=0.24, min=0.05, unit="LENGTH")
    scribe_depth: FloatProperty(
        name="Scribe Depth",
        description="How deeply each log settles into the one below",
        default=0.06,
        min=0.0,
        unit="LENGTH",
    )
    scribe_gap: FloatProperty(
        name="Scribe Gap",
        description="Clearance left in the groove so faces do not coincide",
        default=0.001,
        min=0.0,
        unit="LENGTH",
    )
    alternate_butt: BoolProperty(
        name="Alternate Butt Ends",
        description=(
            "Reverse every other log so a butt always beds onto a top. Keeps "
            "the wall rising level and the groove depth constant"
        ),
        default=True,
    )
    sill_flat: BoolProperty(
        name="Sawn Sill Logs",
        description=(
            "Saw the bottom courses flat where they bed on the foundation. "
            "The first course becomes a half log, the second a full log with a "
            "shallower flat, so all four bottom logs bear on the piles along "
            "their whole length instead of touching only at the butt"
        ),
        default=True,
    )
    flatten_top: BoolProperty(
        name="Flatten Top Course",
        description="Cut a level bearing surface on the uppermost log of each wall",
        default=False,
    )

    # floor
    floor_type: EnumProperty(
        name="Floor",
        items=[
            ("NONE", "None", "No floor structure"),
            (
                "NOTCHED",
                "Notched Logs",
                (
                    "Round joists notched into the sills exactly as an "
                    "internal wall is, flattened on top to carry the boards"
                ),
            ),
            (
                "BRACKET",
                "Joists on Brackets",
                "Sawn rectangular joists carried on brackets fixed to the sills",
            ),
        ],
        default="NOTCHED",
    )
    joist_axis: EnumProperty(
        name="Joist Direction",
        items=[
            ("Y", "Along Y", "Joists span the width, bearing on the X walls"),
            ("X", "Along X", "Joists span the length, bearing on the Y walls"),
        ],
        default="Y",
    )
    joist_spacing: FloatProperty(
        name="Joist Spacing",
        description=(
            "Target centre-to-centre spacing. The real spacing divides the "
            "span evenly, so it comes out at or below this"
        ),
        default=0.6,
        min=0.15,
        unit="LENGTH",
    )
    joist_diameter: FloatProperty(
        name="Joist Diameter", default=0.20, min=0.05, unit="LENGTH"
    )
    log_beam_offset: FloatProperty(
        name="Log Offset",
        description="Centre of the outermost floor log, measured in from the wall",
        default=0.5,
        min=0.0,
        unit="LENGTH",
    )
    log_beam_spacing: FloatProperty(
        name="Log Spacing",
        description=(
            "Centre to centre for the floor logs. The bays either side of the "
            "outermost pair take up whatever does not divide evenly"
        ),
        default=2.0,
        min=0.3,
        unit="LENGTH",
    )
    # roof
    add_roof: BoolProperty(
        name="Roof Frame",
        description="Carry the gables up in logs and span a ridge and purlins between them",
        default=True,
    )
    ridge_axis: EnumProperty(
        name="Ridge",
        items=[
            ("X", "Along X", "Ridge runs the length; the gables are the Y walls"),
            ("Y", "Along Y", "Ridge runs the width; the gables are the X walls"),
        ],
        default="X",
    )
    roof_pitch: FloatProperty(
        name="Pitch",
        description="Roof angle in degrees. 45 puts the ridge half the span above the wall head",
        default=45.0,
        min=10.0,
        max=80.0,
    )
    ridge_diameter: FloatProperty(
        name="Ridge Diameter", default=0.26, min=0.05, unit="LENGTH"
    )
    purlin_diameter: FloatProperty(
        name="Purlin Diameter", default=0.20, min=0.05, unit="LENGTH"
    )
    purlin_spacing: FloatProperty(
        name="Purlin Spacing",
        description="Centre to centre down the rafter line, measured along the slope",
        default=1.2,
        min=0.2,
        unit="LENGTH",
    )
    add_roof_deck: BoolProperty(
        name="Roof Deck",
        description=(
            "Lay plywood sheathing on the ridge and purlins, from the ridge "
            "down to where the wall's own rake cut ends. Joints staggered "
            "the same way as the floor and ceiling"
        ),
        default=True,
    )
    add_rafters: BoolProperty(
        name="Rafters",
        description="Lay rafters on top of the roof deck, running down the slope",
        default=True,
    )
    rafter_diameter: FloatProperty(
        name="Rafter Diameter",
        description="The round log a rafter is milled from before its faces are sawn flat",
        default=0.20,
        min=0.05,
        unit="LENGTH",
    )
    rafter_height: FloatProperty(
        name="Rafter Height",
        description="Flat to flat, once the top and bottom are sawn - the same on every rafter",
        default=0.14,
        min=0.02,
        unit="LENGTH",
    )
    rafter_spacing: FloatProperty(
        name="Rafter Spacing",
        description=(
            "On-centre spacing away from the two gable ends, where the "
            "outermost rafter sits flush with the overhang"
        ),
        default=0.6,
        min=0.15,
        unit="LENGTH",
    )
    noggin_thickness: FloatProperty(
        name="Noggin Thickness",
        description="Sawn board, standing on edge across the full rafter height",
        default=0.025,
        min=0.01,
        unit="LENGTH",
    )
    add_roof_sheathing2: BoolProperty(
        name="Second Sheathing Layer",
        description=(
            "A second plywood layer on top of the rafters, using the same "
            "sheet size and thickness as the deck below"
        ),
        default=True,
    )
    add_roof_laths: BoolProperty(
        name="Ventilation Laths",
        description="Battens on top of the second sheathing, one per rafter, opening a ventilated air gap under the covering",
        default=True,
    )
    lath_width: FloatProperty(
        name="Lath Width", default=0.05, min=0.02, unit="LENGTH"
    )
    lath_thickness: FloatProperty(
        name="Lath Thickness", default=0.025, min=0.01, unit="LENGTH"
    )
    roof_covering: EnumProperty(
        name="Roof Covering",
        items=[
            ("NONE", "None", "No covering - laths left exposed"),
            ("SHINGLES", "Wooden Shingles", "Overlapping wooden shingle courses"),
            ("STONE", "Stone", "Overlapping stone slate courses"),
            ("METAL", "Corrugated Metal", "Corrugated metal panels, one per slope"),
        ],
        default="STONE",
    )
    covering_tile_width: FloatProperty(
        name="Tile Width",
        description="Shingle or stone tile width, shared by both - shrink it for shingles, grow it for stone",
        default=0.20,
        min=0.05,
        unit="LENGTH",
    )
    covering_tile_length: FloatProperty(
        name="Tile Length",
        description="Full length of one tile, up the slope - long enough to reach a good two courses' worth past its own counter-batten",
        default=0.45,
        min=0.05,
        unit="LENGTH",
    )
    covering_exposure: FloatProperty(
        name="Exposure",
        description=(
            "Target counter-batten gauge - how much of each course shows "
            "before the next one up laps over it. Nudged so a whole number "
            "of courses fits the slope. Real double-lap slating keeps this "
            "a bit over half the tile length, so only two courses ever "
            "cover the same spot, not three or more"
        ),
        default=0.20,
        min=0.02,
        unit="LENGTH",
    )
    covering_tile_thickness: FloatProperty(
        name="Tile Thickness",
        description=(
            "Also sets how steeply each tile tilts: it beds on its batten "
            "at the head and one thickness up on the course below one gauge "
            "further down, so the tilt is thickness divided by exposure"
        ),
        default=0.03,
        min=0.003,
        unit="LENGTH",
    )
    covering_corrugation_pitch: FloatProperty(
        name="Corrugation Pitch",
        description="Centre to centre between ribs, across the panel",
        default=0.15,
        min=0.02,
        unit="LENGTH",
    )
    covering_corrugation_depth: FloatProperty(
        name="Corrugation Depth",
        description="Peak to mean height of each rib",
        default=0.02,
        min=0.002,
        unit="LENGTH",
    )
    covering_metal_thickness: FloatProperty(
        name="Metal Thickness", default=0.003, min=0.0005, unit="LENGTH"
    )
    add_counter_battens: BoolProperty(
        name="Counter-Battens",
        description=(
            "Battens across the slope, on top of the ventilation laths, at "
            "the covering's own exposure gauge - what wooden shingles or "
            "stone tiles are actually fastened to. Shingles and stone only; "
            "metal panels fix straight through to the laths"
        ),
        default=True,
    )
    counter_batten_width: FloatProperty(
        name="Counter-Batten Width", default=0.04, min=0.02, unit="LENGTH"
    )
    counter_batten_thickness: FloatProperty(
        name="Counter-Batten Thickness", default=0.02, min=0.01, unit="LENGTH"
    )
    add_ridge_cap: BoolProperty(
        name="Ridge Cap",
        description=(
            "Cap the ridge: wood for wooden shingles, metal for stone or "
            "corrugated metal"
        ),
        default=True,
    )
    ridge_cap_width: FloatProperty(
        name="Ridge Cap Width",
        description="How far the cap reaches down each slope from the peak",
        default=0.25,
        min=0.05,
        unit="LENGTH",
    )
    ridge_cap_metal_thickness: FloatProperty(
        name="Metal Cap Thickness", default=0.003, min=0.0005, unit="LENGTH"
    )
    ridge_cap_wood_thickness: FloatProperty(
        name="Wood Cap Thickness",
        description="Split into two halves so each joint can be a half-lap",
        default=0.03,
        min=0.01,
        unit="LENGTH",
    )
    ridge_cap_piece_length: FloatProperty(
        name="Wood Cap Piece Length",
        description="Length of one wooden cap board before the next overlaps it",
        default=3.0,
        min=0.5,
        unit="LENGTH",
    )
    ridge_cap_lap: FloatProperty(
        name="Wood Cap Overlap",
        description="How far each board's half-lap runs under the next one",
        default=0.10,
        min=0.02,
        unit="LENGTH",
    )
    add_ridge_vents: BoolProperty(
        name="Ridge Vents",
        description="Slots through the cap, letting the air from under the covering out",
        default=True,
    )
    ridge_vent_pitch: FloatProperty(
        name="Vent Spacing",
        description="Centre to centre along the ridge",
        default=0.5,
        min=0.1,
        unit="LENGTH",
    )
    ridge_vent_length: FloatProperty(
        name="Vent Length", default=0.10, min=0.02, unit="LENGTH"
    )
    ridge_vent_width: FloatProperty(
        name="Vent Width", default=0.02, min=0.005, unit="LENGTH"
    )

    joist_overhang: FloatProperty(
        name="Tie-in Overhang",
        description=(
            "How far each floor beam projects past the wall axis it notches "
            "into. Keep it below the sill's radius and the end stays buried "
            "in the sill instead of breaking out through the wall"
        ),
        default=0.20,
        min=0.0,
        unit="LENGTH",
    )
    joist_flat_share: FloatProperty(
        name="Top Flat",
        description=(
            "How much of the joist's diameter is sawn off the top to carry "
            "the boards. A quarter leaves a beam; a half leaves a plank"
        ),
        default=0.25,
        min=0.0,
        max=0.45,
    )
    add_deck: BoolProperty(
        name="Plywood Deck",
        description="Lay sheets over the structure, joints staggered",
        default=True,
    )
    add_ceiling: BoolProperty(
        name="Ceiling",
        description=(
            "Build the same structure again at the head of the walls, in "
            "whichever style is selected above"
        ),
        default=True,
    )
    sheet_width: FloatProperty(
        name="Sheet Width",
        description="Across the beams",
        default=1.2,
        min=0.2,
        unit="LENGTH",
    )
    sheet_length: FloatProperty(
        name="Sheet Length",
        description="Along the beams. Alternate rows shift by half of this",
        default=2.0,
        min=0.2,
        unit="LENGTH",
    )
    sheet_thickness: FloatProperty(
        name="Sheet Thickness", default=0.022, min=0.003, unit="LENGTH"
    )
    joist_wall_gap: FloatProperty(
        name="Wall Gap",
        description=(
            "Clear gap between the outermost joist and the wall it runs "
            "alongside. The bays either side of it take up whatever the "
            "spacing does not divide evenly"
        ),
        default=0.10,
        min=0.0,
        unit="LENGTH",
    )
    joist_width: FloatProperty(name="Joist Width", default=0.10, min=0.02, unit="LENGTH")
    joist_depth: FloatProperty(name="Joist Depth", default=0.20, min=0.05, unit="LENGTH")
    bracket_thickness: FloatProperty(
        name="Bracket Thickness", default=0.012, min=0.002, unit="LENGTH"
    )
    bracket_mill: FloatProperty(
        name="Milled Face",
        description=(
            "How much is milled off the inner face of the sills the brackets "
            "bolt to, along their whole length. Everything else - bracket, "
            "joist ends - is measured out from that face"
        ),
        default=0.04,
        min=0.0,
        unit="LENGTH",
    )
    top_flat_depth: FloatProperty(
        name="Top Flat Depth",
        description="How far down from the crown the flat is cut",
        default=0.04,
        min=0.0,
        unit="LENGTH",
    )

    # natural variation
    seed: IntProperty(name="Seed", default=1, min=0)
    bow: FloatProperty(
        name="Bow",
        description="Sideways sweep as a fraction of log length",
        default=0.005,
        min=0.0,
        max=0.1,
        precision=4,
    )
    bow_vertical: FloatProperty(
        name="Vertical Bow Share",
        description=(
            "How much of the bow may run up and down. Builders roll logs so "
            "the sweep lies sideways; 0 keeps every bow horizontal"
        ),
        default=0.25,
        min=0.0,
        max=1.0,
    )
    radial_jitter: FloatProperty(
        name="Thickness Variation",
        description="Fractional wobble in log radius along its length",
        default=0.03,
        min=0.0,
        max=0.4,
        precision=3,
    )
    knot_density: FloatProperty(
        name="Knots per Metre",
        description=(
            "How thickly knots are scattered along each log. 0 leaves the "
            "timber clean. They appear only where nothing has been cut"
        ),
        default=0.6,
        min=0.0,
        max=8.0,
    )
    knot_size: FloatProperty(
        name="Knot Size",
        description="How far across the log's surface a knot spreads",
        default=0.06,
        min=0.005,
        unit="LENGTH",
    )
    knot_rise: FloatProperty(
        name="Knot Rise",
        description="How far a knot stands proud of the barrel",
        default=0.008,
        min=0.0,
        unit="LENGTH",
    )

    # resolution
    axial_segments: IntProperty(
        name="Axial Segments",
        description=(
            "Baseline stations along the log. Only has to resolve the bow, "
            "since crossings get their own dense stations"
        ),
        default=20,
        min=6,
        max=200,
    )
    radial_segments: IntProperty(
        name="Radial Segments",
        description=(
            "Vertices around the log. Half go over the barrel, half along the "
            "floor, where the cut rims claim a few"
        ),
        default=28,
        min=6,
        max=96,
    )
    notch_refine: IntProperty(
        name="Notch Refinement",
        description=(
            "Extra stations packed into each crossing. Raise this if saddle "
            "notches look faceted along the log; it costs nothing elsewhere"
        ),
        default=16,
        min=4,
        max=80,
    )
    shade_smooth: BoolProperty(
        name="Smooth Shading",
        description="Cosmetic only, the mesh is unchanged",
        default=True,
    )

    # internal walls
    internal_walls: CollectionProperty(type=LOGCABIN_InternalWall)
    active_internal: IntProperty(default=0)
    internal_overhang: FloatProperty(
        name="Tie-in Overhang",
        description="How far internal logs project past the external wall they notch into",
        default=0.15,
        min=0.0,
        unit="LENGTH",
    )


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------


class LOGCABIN_OT_add_internal(bpy.types.Operator):
    bl_idname = "logcabin.add_internal"
    bl_label = "Add Internal Wall"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.log_cabin
        wall = props.internal_walls.add()
        wall.height = props.wall_height
        props.active_internal = len(props.internal_walls) - 1
        return {"FINISHED"}


class LOGCABIN_OT_remove_internal(bpy.types.Operator):
    bl_idname = "logcabin.remove_internal"
    bl_label = "Remove Internal Wall"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.log_cabin.internal_walls) > 0

    def execute(self, context):
        props = context.scene.log_cabin
        props.internal_walls.remove(props.active_internal)
        props.active_internal = max(0, props.active_internal - 1)
        return {"FINISHED"}


class LOGCABIN_OT_generate(bpy.types.Operator):
    bl_idname = "logcabin.generate"
    bl_label = "Generate Cabin"
    bl_description = "Rebuild all logs and piles from the current settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.log_cabin

        try:
            records, piles, joists, boxes = generate_cabin(props)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        coll = _get_collection(context)

        for index, (log, wall_name, course) in enumerate(records):
            creases = []
            verts, faces = build_log_mesh(
                log,
                props.axial_segments,
                props.radial_segments,
                props.scribe_gap,
                props.notch_refine,
                creases,
            )
            obj = _make_object(
                coll,
                f"Log_{wall_name}_{course:02d}_{index:03d}",
                verts,
                faces,
                "log",
                wall_name,
                course,
            )
            _apply_shading(obj, props.shade_smooth, creases)

        for index, (x, y) in enumerate(piles):
            verts, faces = build_cylinder_mesh(
                (x, y),
                props.pile_diameter / 2.0,
                -props.pile_height,
                0.0,
            )
            obj = _make_object(coll, f"Pile_{index:03d}", verts, faces, "pile")
            _apply_shading(obj, props.shade_smooth)

        for log, name, kind in joists:
            creases = []
            verts, faces = build_log_mesh(
                log,
                props.axial_segments,
                props.radial_segments,
                props.scribe_gap,
                props.notch_refine,
                creases,
            )
            obj = _make_object(coll, name, verts, faces, kind)
            _apply_shading(obj, props.shade_smooth, creases)

        for name, verts, faces, kind in boxes:
            # Sawn timber, brackets and sheets are genuinely faceted.
            obj = _make_object(coll, name, verts, faces, kind)
            _apply_shading(obj, False)

        tally = {}
        for entry in boxes:
            tally[entry[3]] = tally.get(entry[3], 0) + 1
        message = (
            f"Built {len(records)} logs, {len(piles)} piles, "
            f"{len(joists) + tally.get('joist', 0)} joists"
        )
        if tally.get("bracket"):
            message += f", {tally['bracket']} bracket parts"
        if tally.get("sheet"):
            message += f", {tally['sheet']} deck sheets"
        self.report({"INFO"}, message + ".")
        return {"FINISHED"}


# --------------------------------------------------------------------------
# IFC export
# --------------------------------------------------------------------------


def _ifc_state():
    """Return (ok, message, module_bundle) describing Bonsai readiness."""
    try:
        import ifcopenshell
        import ifcopenshell.api
        import ifcopenshell.util.unit
        from bonsai import tool
    except ImportError as exc:
        return False, f"Bonsai / IFCOpenShell not importable: {exc}", None

    ifc = tool.Ifc.get()
    if ifc is None:
        return (
            False,
            "No IFC project. Create one first via Bonsai > Project > New Project.",
            None,
        )
    if not ifc.by_type("IfcBuildingStorey"):
        return False, "IFC project has no IfcBuildingStorey to contain the cabin.", None

    return True, "", (ifcopenshell, tool, ifc)


def _run_api(ifcopenshell, action, ifc, plural, value, **kwargs):
    """ifcopenshell.api moved from product= to products= mid-life. Try both."""
    try:
        return ifcopenshell.api.run(action, ifc, **{plural: [value]}, **kwargs)
    except TypeError:
        return ifcopenshell.api.run(action, ifc, **{plural[:-1]: value}, **kwargs)


def _body_context(ifcopenshell, ifc):
    for ctx in ifc.by_type("IfcGeometricRepresentationSubContext"):
        if ctx.ContextIdentifier == "Body":
            return ctx
    parent = None
    for ctx in ifc.by_type("IfcGeometricRepresentationContext", include_subtypes=False):
        if ctx.ContextType == "Model":
            parent = ctx
            break
    if parent is None:
        parent = ifcopenshell.api.run("context.add_context", ifc, context_type="Model")
    return ifcopenshell.api.run(
        "context.add_context",
        ifc,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=parent,
    )


def _tessellate(ifc, obj, unit_scale):
    mesh = obj.data
    matrix = obj.matrix_world

    coords = []
    for vertex in mesh.vertices:
        world = matrix @ vertex.co
        coords.append(
            (world.x / unit_scale, world.y / unit_scale, world.z / unit_scale)
        )

    triangles = []
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        for i in range(1, len(indices) - 1):
            triangles.append((indices[0] + 1, indices[i] + 1, indices[i + 1] + 1))

    point_list = ifc.create_entity("IfcCartesianPointList3D", CoordList=coords)
    return ifc.create_entity(
        "IfcTriangulatedFaceSet",
        Coordinates=point_list,
        CoordIndex=triangles,
        Closed=True,
    )


def _assign_shape(ifc, element, item, body):
    representation = ifc.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body,
        RepresentationIdentifier="Body",
        RepresentationType="Tessellation",
        Items=[item],
    )
    element.Representation = ifc.create_entity(
        "IfcProductDefinitionShape", Representations=[representation]
    )


def _identity_placement(ifc, relative_to=None):
    point = ifc.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    axes = ifc.create_entity("IfcAxis2Placement3D", Location=point)
    placement = ifc.create_entity("IfcLocalPlacement", RelativePlacement=axes)
    if relative_to is not None:
        placement.PlacementRelTo = relative_to
    return placement


class LOGCABIN_OT_to_ifc(bpy.types.Operator):
    bl_idname = "logcabin.to_ifc"
    bl_label = "Convert to IFC"
    bl_description = (
        "Create IfcWall / IfcMember / IfcFooting entities in the open Bonsai project"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        ok, message, bundle = _ifc_state()
        if not ok:
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        ifcopenshell, _tool, ifc = bundle

        coll = bpy.data.collections.get(COLLECTION_NAME)
        if coll is None or not coll.objects:
            self.report({"ERROR"}, "Nothing to convert. Generate the cabin first.")
            return {"CANCELLED"}

        unit_scale = ifcopenshell.util.unit.calculate_unit_scale(ifc)
        storey = ifc.by_type("IfcBuildingStorey")[0]
        body = _body_context(ifcopenshell, ifc)
        storey_placement = getattr(storey, "ObjectPlacement", None)

        walls = {}
        wall_count = 0
        member_count = 0
        footing_count = 0
        floor_count = 0

        for obj in sorted(coll.objects, key=lambda o: o.name):
            kind = obj.get(TYPE_KEY)

            if kind == "log":
                wall_name = obj.get(WALL_KEY, "Wall")
                wall = walls.get(wall_name)
                if wall is None:
                    wall = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcWall",
                        predefined_type="ELEMENTEDWALL",
                        name=wall_name,
                    )
                    wall.ObjectPlacement = _identity_placement(ifc, storey_placement)
                    _run_api(
                        ifcopenshell,
                        "spatial.assign_container",
                        ifc,
                        "products",
                        wall,
                        relating_structure=storey,
                    )
                    walls[wall_name] = wall
                    wall_count += 1

                member = ifcopenshell.api.run(
                    "root.create_entity", ifc, ifc_class="IfcMember", name=obj.name
                )
                member.ObjectPlacement = _identity_placement(ifc, storey_placement)
                _assign_shape(ifc, member, _tessellate(ifc, obj, unit_scale), body)
                _run_api(
                    ifcopenshell,
                    "aggregate.assign_object",
                    ifc,
                    "products",
                    member,
                    relating_object=wall,
                )
                member_count += 1

            elif kind == "pile":
                footing = ifcopenshell.api.run(
                    "root.create_entity",
                    ifc,
                    ifc_class="IfcFooting",
                    predefined_type="PILE_FOUNDATION",
                    name=obj.name,
                )
                footing.ObjectPlacement = _identity_placement(ifc, storey_placement)
                _assign_shape(ifc, footing, _tessellate(ifc, obj, unit_scale), body)
                _run_api(
                    ifcopenshell,
                    "spatial.assign_container",
                    ifc,
                    "products",
                    footing,
                    relating_structure=storey,
                )
                footing_count += 1

            elif kind in (
                "joist", "bracket", "sheet", "purlin", "rafter", "lath", "covering",
            ):
                if kind == "purlin":
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcBeam",
                        predefined_type="BEAM",
                        name=obj.name,
                    )
                elif kind == "rafter":
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcBeam",
                        predefined_type="RAFTER",
                        name=obj.name,
                    )
                elif kind == "joist":
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcBeam",
                        predefined_type="JOIST",
                        name=obj.name,
                    )
                elif kind == "sheet":
                    # IfcPlate is the planar element of constant thickness,
                    # which is what a sheet good is.
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcPlate",
                        predefined_type="SHEET",
                        name=obj.name,
                    )
                elif kind == "lath":
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcMember",
                        predefined_type="MEMBER",
                        name=obj.name,
                    )
                elif kind == "covering":
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcCovering",
                        predefined_type="ROOFING",
                        name=obj.name,
                    )
                else:
                    # IfcDiscreteAccessory is the fitting for a fixing that is
                    # not itself structure - brackets, shoes, anchor plates.
                    element = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc,
                        ifc_class="IfcDiscreteAccessory",
                        name=obj.name,
                    )
                element.ObjectPlacement = _identity_placement(ifc, storey_placement)
                _assign_shape(ifc, element, _tessellate(ifc, obj, unit_scale), body)
                _run_api(
                    ifcopenshell,
                    "spatial.assign_container",
                    ifc,
                    "products",
                    element,
                    relating_structure=storey,
                )
                floor_count += 1

        self.report(
            {"INFO"},
            f"Created {wall_count} walls, {member_count} logs, "
            f"{footing_count} footings, {floor_count} floor parts. "
            f"Save via Bonsai to write the IFC file.",
        )
        return {"FINISHED"}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------


class LOGCABIN_UL_internal(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        row.label(text=f"{index + 1}")
        row.prop(item, "axis", text="")
        row.prop(item, "position", text="Pos")
        row.prop(item, "height", text="H")


def _wrap(text, width):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


class LOGCABIN_PT_panel(bpy.types.Panel):
    bl_label = "Log Cabin"
    bl_idname = "LOGCABIN_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Log Cabin"

    def draw(self, context):
        layout = self.layout
        props = context.scene.log_cabin

        box = layout.box()
        box.label(text="Footprint", icon="MESH_PLANE")
        box.prop(props, "length")
        box.prop(props, "width")
        box.prop(props, "wall_height")
        box.prop(props, "overhang")

        box = layout.box()
        box.label(text="Foundation", icon="SNAP_VERTEX")
        box.prop(props, "max_span")
        box.prop(props, "pile_diameter")
        box.prop(props, "pile_height")

        box = layout.box()
        box.label(text="Logs", icon="MOD_SKIN")
        box.prop(props, "butt_diameter")
        box.prop(props, "top_diameter")
        box.prop(props, "scribe_depth")
        box.prop(props, "scribe_gap")
        box.prop(props, "alternate_butt")
        box.prop(props, "sill_flat")
        box.prop(props, "flatten_top")
        if props.flatten_top:
            box.prop(props, "top_flat_depth")

        box = layout.box()
        box.label(text="Floor", icon="MESH_GRID")
        box.prop(props, "floor_type", expand=True)
        if props.floor_type != "NONE":
            box.prop(props, "joist_axis", expand=True)
            if props.floor_type == "NOTCHED":
                box.prop(props, "joist_diameter")
                box.prop(props, "joist_overhang")
                box.prop(props, "joist_flat_share")
                box.prop(props, "log_beam_offset")
                box.prop(props, "log_beam_spacing")
                box.separator()
                box.label(text="Beams across the logs", icon="MESH_CUBE")
                box.prop(props, "joist_width")
                box.prop(props, "joist_depth")
                box.prop(props, "joist_spacing")
                box.prop(props, "joist_wall_gap")
            else:
                box.prop(props, "joist_spacing")
                box.prop(props, "joist_wall_gap")
                box.prop(props, "joist_width")
                box.prop(props, "joist_depth")
                box.prop(props, "bracket_thickness")
                box.prop(props, "bracket_mill")

            box.separator()
            box.prop(props, "add_ceiling")
            box.prop(props, "add_deck")
            if props.add_deck:
                box.prop(props, "sheet_width")
                box.prop(props, "sheet_length")
                box.prop(props, "sheet_thickness")

        box = layout.box()
        box.label(text="Roof", icon="MESH_CONE")
        box.prop(props, "add_roof")
        if props.add_roof:
            box.prop(props, "ridge_axis", expand=True)
            box.prop(props, "roof_pitch")
            box.prop(props, "ridge_diameter")
            box.prop(props, "purlin_diameter")
            box.prop(props, "purlin_spacing")
            box.prop(props, "add_roof_deck")
            box.prop(props, "add_rafters")
            if props.add_rafters:
                box.prop(props, "rafter_diameter")
                box.prop(props, "rafter_height")
                box.prop(props, "rafter_spacing")
                box.prop(props, "noggin_thickness")
                box.prop(props, "add_roof_sheathing2")
                box.prop(props, "add_roof_laths")
                if props.add_roof_laths:
                    box.prop(props, "lath_width")
                    box.prop(props, "lath_thickness")
                box.prop(props, "roof_covering")
                if props.roof_covering in ("SHINGLES", "STONE"):
                    box.prop(props, "covering_tile_width")
                    box.prop(props, "covering_tile_length")
                    box.prop(props, "covering_exposure")
                    box.prop(props, "covering_tile_thickness")
                    box.prop(props, "add_counter_battens")
                    if props.add_counter_battens:
                        box.prop(props, "counter_batten_width")
                        box.prop(props, "counter_batten_thickness")
                elif props.roof_covering == "METAL":
                    box.prop(props, "covering_corrugation_pitch")
                    box.prop(props, "covering_corrugation_depth")
                    box.prop(props, "covering_metal_thickness")
                if props.roof_covering != "NONE":
                    box.prop(props, "add_ridge_cap")
                    if props.add_ridge_cap:
                        box.prop(props, "ridge_cap_width")
                        if props.roof_covering == "SHINGLES":
                            box.prop(props, "ridge_cap_wood_thickness")
                            box.prop(props, "ridge_cap_piece_length")
                            box.prop(props, "ridge_cap_lap")
                        else:
                            box.prop(props, "ridge_cap_metal_thickness")
                        box.prop(props, "add_ridge_vents")
                        if props.add_ridge_vents:
                            box.prop(props, "ridge_vent_pitch")
                            box.prop(props, "ridge_vent_length")
                            box.prop(props, "ridge_vent_width")

        box = layout.box()
        box.label(text="Natural Variation", icon="RNDCURVE")
        box.prop(props, "seed")
        box.prop(props, "bow")
        box.prop(props, "bow_vertical")
        box.prop(props, "radial_jitter")
        box.prop(props, "knot_density")
        if props.knot_density > 0.0:
            box.prop(props, "knot_size")
            box.prop(props, "knot_rise")

        box = layout.box()
        box.label(text="Internal Walls", icon="MOD_BUILD")
        row = box.row()
        row.template_list(
            "LOGCABIN_UL_internal",
            "",
            props,
            "internal_walls",
            props,
            "active_internal",
            rows=2,
        )
        col = row.column(align=True)
        col.operator("logcabin.add_internal", icon="ADD", text="")
        col.operator("logcabin.remove_internal", icon="REMOVE", text="")
        box.prop(props, "internal_overhang")

        box = layout.box()
        box.label(text="Resolution", icon="MESH_GRID")
        box.prop(props, "axial_segments")
        box.prop(props, "radial_segments")
        box.prop(props, "notch_refine")
        box.prop(props, "shade_smooth")

        layout.separator()
        layout.operator("logcabin.generate", icon="MOD_BUILD")

        ok, message, _ = _ifc_state()
        if not ok:
            warn = layout.box()
            warn.alert = True
            warn.label(text="IFC export unavailable:", icon="ERROR")
            for line in _wrap(message, 34):
                warn.label(text=line)
        layout.operator("logcabin.to_ifc", icon="EXPORT")


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


CLASSES = (
    LOGCABIN_InternalWall,
    LOGCABIN_Props,
    LOGCABIN_OT_add_internal,
    LOGCABIN_OT_remove_internal,
    LOGCABIN_OT_generate,
    LOGCABIN_OT_to_ifc,
    LOGCABIN_UL_internal,
    LOGCABIN_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.log_cabin = PointerProperty(type=LOGCABIN_Props)


def unregister():
    del bpy.types.Scene.log_cabin
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    # Re-running from the Text Editor should replace the previous
    # registration rather than erroring on duplicate classes.
    if hasattr(bpy.types.Scene, "log_cabin"):
        unregister()
    register()
