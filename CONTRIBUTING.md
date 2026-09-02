# Contributing to IFC_Log-Cabin

## Setup (once)

```
git clone https://github.com/DaBje/IFC_Log-Cabin.git
cd IFC_Log-Cabin
```

## Workflow for every change

### 1. Start from an up-to-date main
```
git checkout main
git pull
```

### 2. Create a branch
```
git checkout -b feature/your-feature-name   # new feature        → minor bump (1.0.x → 1.1.0)
git checkout -b fix/what-you-are-fixing     # bug fix or UI change → patch bump (1.0.0 → 1.0.1)
```

### 3. Make your changes and test in Blender

### 4. Bump the version in `bl_info` and commit
```
git add IFC_Log_Cabin.py
git commit -m "Short description of what changed and why"
```

### 5. Push the branch
First push on a new branch requires setting the upstream:
```
git push --set-upstream origin your-branch-name
```
After that, plain `git push` works for all subsequent pushes on the same branch.

### 6. Merge to main
```
git checkout main
git merge your-branch-name
git push
```
Or open a Pull Request on github.com/DaBje/IFC_Log-Cabin if you want a review before merging.

### 7. Tag the release (repo owner only)
Both commands use the same version number — `git tag` creates the tag locally, `git push origin` sends it to GitHub:
```
git tag v1.0.0
git push origin v1.0.0
```

---

## Version numbers (inside `bl_info`)

| Change type | Example | When |
|---|---|---|
| Bug fix | 1.0.0 → 1.0.1 | Correcting wrong values, crashes |
| UI change | 1.0.0 → 1.0.1 | Layout, labels, dropdown width — no new behaviour |
| New feature | 1.0.x → 1.1.0 | New behaviour, new UI element |
| Major revision | 1.x.x → 2.0.0 | Large changes |

## Working with unfinished changes across branches

Uncommitted changes follow you when you switch branches (if there is no conflict). Use `git stash` to park them safely before switching.

### Basic workflow
```
git stash                  # save changes, clean working dir
git checkout main          # switch freely
git checkout feature/xyz   # come back later
git stash pop              # restore your changes
```

### Working with multiple stashes
Always name them so you can tell them apart:
```
git stash push -m "descriptive name"
```

List, inspect, and restore by index:
```
git stash list             # see all stashes with indices
git stash show stash@{1}   # inspect a specific one
git stash pop stash@{1}    # restore a specific one
git stash drop stash@{1}   # delete one without applying
```

`stash@{0}` is always the most recent. Indices shift down after a pop or drop.

---

## Useful commands

| Command | What it does |
|---|---|
| `git branch` | Show all branches, `*` = current |
| `git status` | Show what has changed since last commit |
| `git log --oneline` | Show commit history |
| `git diff` | Show exact line-by-line changes not yet committed |
| `git checkout main` | Switch back to main |
| `git pull` | Download latest changes from GitHub |
| `git stash` | Save uncommitted changes and clean working dir |
| `git stash pop` | Restore most recent stash |
| `git stash list` | See all saved stashes |
