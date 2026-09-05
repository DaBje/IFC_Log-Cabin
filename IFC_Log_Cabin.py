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
    "version": (0, 1, 1),
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
      notches - list of (distance along axis, drop, radius of crossing log)
      flat_z  - world height to flatten the top to, or None
    """

    def __init__(self, p0, p1, r_butt, r_top, rng, bow, radial_jitter, bow_vertical=0.25):
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

        self.groove = None
        self.notches = []
        self.flat_z = None
        self.floor_z = None

    def axis_point(self, t):
        return self.p0.lerp(self.p1, t) + self.bow_vec * math.sin(math.pi * t)

    def radius(self, t):
        base = _lerp(self.r0, self.r1, t)
        wobble = 1.0 + self.jitter_amp * math.sin(
            self.jitter_phase + self.jitter_freq * math.tau * t
        )
        return base * wobble


# --------------------------------------------------------------------------
# cross-section profile
# --------------------------------------------------------------------------


def _floor_samples(r, breaks, count):
    """`count` positions across the log, with every break landing exactly.

    A cut's rim - where it meets the barrel - is a crease. Sampling straight
    through it leaves the crease running diagonally across a quad, which is
    what makes an otherwise correct notch look chewed. Placing a vertex on the
    rim and sharing the rest out by width keeps the count fixed and the rim
    crisp.
    """
    if count <= 0:
        return []

    inner = sorted(b for b in breaks if -r + 1e-6 < b < r - 1e-6)
    if len(inner) >= count:
        return inner[:count]

    knots = [-r] + inner + [r]
    widths = [knots[i + 1] - knots[i] for i in range(len(knots) - 1)]

    shares = [0] * len(widths)
    for _ in range(count - len(inner)):
        widest = max(
            range(len(widths)), key=lambda i: widths[i] / (shares[i] + 1)
        )
        shares[widest] += 1

    points = list(inner)
    for i, share in enumerate(shares):
        low, high = knots[i], knots[i + 1]
        for k in range(1, share + 1):
            points.append(low + (high - low) * k / (share + 1))
    return sorted(points)


def _ring_points(log, t, along, gap, n_up, n_low):
    """The log's outline at one station, as (a, b) in its own frame.

    a runs across the log, b runs up it. Fixed point count, fixed ordering:
    over the top from a=+r to a=-r, then back along the floor. Because the
    topology never changes between stations, rings stitch into clean quads
    with no resampling - which is what stops the surface smearing.
    """
    r = log.radius(t)
    centre = log.axis_point(t)

    ceiling = None if log.flat_z is None else log.flat_z - centre.z

    # Saddle notches. A perpendicular log removes a horizontal slab, so its
    # contribution to the floor is a constant, not a curve.
    line_top = None
    for offset, drop, radius in log.notches:
        reach = radius + gap
        span = reach * reach - (along - offset) ** 2
        if span <= 0.0:
            continue  # this station clears the crossing
        top = math.sqrt(span) - drop
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
        return b

    # Rims: where each cut meets the barrel. Both are closed form.
    notch_rim = None
    if line_top is not None and -r < line_top < r:
        # A horizontal cut meets the circle at a = +/-sqrt(r^2 - top^2).
        notch_rim = math.sqrt(r * r - line_top * line_top)

    groove_rim = None
    if groove is not None:
        # Circle meets circle: with A = sqrt(rg^2 - u), B = sqrt(r^2 - u) and
        # A + B = drop, then A - B = (rg^2 - r^2)/drop, so A follows directly.
        drop, r_groove = groove
        if drop > 1e-9:
            arc = (drop + (r_groove * r_groove - r * r) / drop) * 0.5
            u = r_groove * r_groove - arc * arc
            if u > 0.0:
                groove_rim = math.sqrt(u)

    # The outermost rim is the section's shoulder, and it is always carried by
    # the sweep's two endpoints - at any cut depth, above or below the axis.
    # Letting it migrate between the sweep and the floor samples, which is
    # what happened while only notches biting above the axis shortened the
    # sweep, hands the rim to a different vertex index from one ring to the
    # next. The strip between two such rings jogs sideways by a vertex, and
    # that is what reads as a staircase along the edge.
    outer = [rim for rim in (notch_rim, groove_rim) if rim is not None]
    a_rim = max(r * 1e-3, min(r, max(outer))) if outer else r

    # Creases strictly inside the section still need vertices of their own.
    # _floor_samples drops anything at the shoulder, so the outer rim falls
    # away here on its own and only the inner one survives.
    breaks = []
    for rim in outer:
        breaks.extend((-rim, rim))
    if groove is not None and line_top is not None:
        # Where the notch takes over from the groove. The floor is the upper
        # of the two, and they cross in a crease running along the notch.
        drop, r_groove = groove
        height = line_top + drop
        if 0.0 < height < r_groove:
            edge = math.sqrt(r_groove * r_groove - height * height)
            breaks.extend((-edge, edge))

    # Sweep the barrel from one shoulder, over the crown, to the other. The
    # shoulder lies on the circle, so atan2 gives its angle directly and the
    # sweep runs past the horizontal whenever the cut sits below the axis.
    phi0 = math.atan2(lower(a_rim), a_rim)
    sweep = math.pi - 2.0 * phi0

    points = []
    for k in range(n_up):
        phi = phi0 + sweep * k / (n_up - 1)
        b = r * math.sin(phi)
        points.append(
            (r * math.cos(phi), b if ceiling is None else min(b, ceiling))
        )
    for a in _floor_samples(a_rim, breaks, n_low):
        # a_rim already keeps the floor below the ceiling; the clamp is only a
        # guard against the outline crossing itself if the two ever meet.
        points.append((a, min(lower(a), upper(a))))
    return points


def _station_params(log, axial, refine):
    """Uniform stations, plus a dense cluster at each notch.

    A notch spans about one log diameter out of several metres. Sampling the
    whole log finely enough to catch it would be wasted everywhere else, and
    sampling it coarsely is what made notches look chiselled.
    """
    params = {round(i / axial, 6) for i in range(axial + 1)}
    if log.length > 1e-9:
        for offset, _drop, radius in log.notches:
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
            # sits at its own axis height right where it runs out, so there is
            # a genuine step there; this makes it a clean edge, not a ramp.
            for edge, outward in ((offset - radius, -1.0), (offset + radius, 1.0)):
                param = (edge + outward * 0.004) / log.length
                if 0.0 <= param <= 1.0:
                    params.add(round(param, 6))

    # Rounding alone can still leave stations a micrometre apart, and a ring
    # pair that close makes zero-area quads that shade black.
    ordered = sorted(params)
    spaced = [ordered[0]]
    for value in ordered[1:]:
        if value - spaced[-1] > 1e-5:
            spaced.append(value)
    if spaced[-1] < 1.0 - 1e-9:
        spaced[-1] = 1.0
    return spaced


def build_log_mesh(log, axial, radial, gap, refine):
    """Return (verts, faces) for one finished log, in world coordinates."""
    n_up = max(3, radial // 2 + 1)
    n_low = max(1, radial - n_up)
    radial = n_up + n_low

    params = _station_params(log, axial, refine)

    verts = []
    for t in params:
        centre = log.axis_point(t)
        ring = _ring_points(log, t, t * log.length, gap, n_up, n_low)
        for a, b in ring:
            verts.append(centre + log.side * a + log.up * b)

    rings = len(params)
    faces = []
    for i in range(rings - 1):
        for j in range(radial):
            j_next = (j + 1) % radial
            faces.append(
                (
                    i * radial + j,
                    (i + 1) * radial + j,
                    (i + 1) * radial + j_next,
                    i * radial + j_next,
                )
            )

    # Cap centres use the ring centroid, not the axis: a deeply notched end
    # ring can leave the axis outside the remaining outline.
    def centroid(base):
        total = Vector((0.0, 0.0, 0.0))
        for j in range(radial):
            total += verts[base + j]
        return total / radial

    last = (rings - 1) * radial
    start_centre = len(verts)
    verts.append(centroid(0))
    end_centre = len(verts)
    verts.append(centroid(last))
    for j in range(radial):
        j_next = (j + 1) % radial
        faces.append((start_centre, j_next, j))
        faces.append((end_centre, last + j, last + j_next))

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
    for course in range(max_courses):
        below = lines_at(course - 1)
        two_below = {index for index, _line in lines_at(course - 2)}
        above = {index for index, _line in lines_at(course + 2)}

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
            )

            # Lateral groove, cut to the log two courses down on the same wall.
            # Its taper runs opposite to this one whenever the ends alternate,
            # so a butt always beds onto a top.
            if wall_index in two_below:
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
                log.notches.append(
                    (
                        offset,
                        half_rise,
                        radius_at(cross, course - 1, cross_index, line.position),
                    )
                )

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

    piles = []
    if props.pile_height > 0.0:
        piles = list(pile_positions(props))

    return records, piles


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


def _apply_shading(obj, smooth):
    """Smooth shading that keeps real creases sharp."""
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
            records, piles = generate_cabin(props)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        coll = _get_collection(context)

        for index, (log, wall_name, course) in enumerate(records):
            verts, faces = build_log_mesh(
                log,
                props.axial_segments,
                props.radial_segments,
                props.scribe_gap,
                props.notch_refine,
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
            _apply_shading(obj, props.shade_smooth)

        for index, (x, y) in enumerate(piles):
            verts, faces = build_cylinder_mesh(
                (x, y),
                props.pile_diameter / 2.0,
                -props.pile_height,
                0.0,
            )
            obj = _make_object(coll, f"Pile_{index:03d}", verts, faces, "pile")
            _apply_shading(obj, props.shade_smooth)

        self.report(
            {"INFO"},
            f"Built {len(records)} logs and {len(piles)} piles.",
        )
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

        self.report(
            {"INFO"},
            f"Created {wall_count} walls, {member_count} logs, "
            f"{footing_count} footings. Save via Bonsai to write the IFC file.",
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
        box.label(text="Natural Variation", icon="RNDCURVE")
        box.prop(props, "seed")
        box.prop(props, "bow")
        box.prop(props, "bow_vertical")
        box.prop(props, "radial_jitter")

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
