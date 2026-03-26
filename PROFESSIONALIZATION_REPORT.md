# Repository professionalization report

**Branch:** `professionalization-cleanup`  
**Repository:** [DesignAInf/DAU-software-tool](https://github.com/DesignAInf/DAU-software-tool)  
**Date:** March 2025

This note summarizes cleanup work for a research-oriented companion repo to *Design for Entropy* (MIT Press). The top-level `README.md` was **not** modified, per project constraints.

## What changed

### Structure

- All prototype trees were moved under **`prototypes/`** with **snake_case** folder names (e.g. `dau_v2_dashboard`, `battery_design_fep`, `joule_active_inference_model`).
- Added **`prototypes/README.md`** as a navigation table for the new layout.

### DAU v2 dashboard — AI-generated static visuals

- Removed the four files **`dashboard_interface1.png` … `dashboard_interface4.png`** from `prototypes/dau_v2_dashboard/` (they were not referenced by the live HTML; the UI already plots via **canvas + JSON** from Flask).
- Added **`dau_project/generate_reference_plots.py`**: runs the same simulation and writes **matplotlib** PNGs under `dau_project/results/`, equivalent to the metrics shown in the dashboard.
- Documented this in **`README.md`**, **`README_DASHBOARD.md`**, and **`dau_project/requirements.txt`**.

**Scope note:** Other image assets elsewhere in the repo (e.g. DAU v3 interface PNGs, Beck–Ramstead figures, Joule result plots) were **not** removed; the task specified **only** the DAU v2 dashboard folder.

### Dependency management

- **`requirements.txt`** at repo root: optional **`ruff`** for linting.
- Per-component **`requirements.txt`** added where missing: `dau_v2`, `dau_v2_dashboard/dau_project`, `dau_v3_dashboard`, `battery_design_fep`, `dau_model_rigid_user`, `dau_model_rigid_designer`, and **comment-only** stubs for the three **Beck–Ramstead** Vite demos (`npm install` / `npm run dev`).
- Existing **`requirements.txt` / `pyproject.toml`** under `designer_artifact_user`, `emergent_conduit_fep`, and `joule_active_inference_model` were left in place (paths updated only by the folder move).

### Code hygiene

- Root **`.gitignore`** for Python caches, virtualenvs, common IDE/OS noise.
- Removed tracked **`__pycache__`** directories from the working tree under the v2 dashboard (they should not be versioned).
- Light cleanup on **`server.py`** (v2) and **`server_v3.py`**: normalized imports, dropped unused `json` import.

### Documentation

- New or expanded **`README.md`** files: `prototypes/dau_v2`, `prototypes/dau_v2_dashboard`, `prototypes/dau_model_rigid_user`, `prototypes/dau_model_rigid_designer`, plus the index under `prototypes/`.
- Long-form docs (e.g. `README_DASHBOARD.md`, battery README) kept; small cross-links where helpful.

### Git

- Intended commit messages (for maintainers to apply locally if the index was locked during automation):
  1. `chore: move prototypes under prototypes/ with consistent names`
  2. `feat(dau-v2-dashboard): drop AI screenshots; add matplotlib reference export`
  3. `chore: add .gitignore, per-prototype requirements, and docs`
  4. `style: tidy Flask server imports`

If your environment reported **`unable to write new index file`**, run `git add` / `git commit` from a clean shell after closing tools that lock `.git/index`.

## Verification checklist

- **No `dashboard_interface*.png`** under `prototypes/dau_v2_dashboard/` (confirm with `git ls-files` / search).
- **Root `README.md`:** unchanged.
- **Run smoke tests** (on a machine with Python 3.10+):
  - `cd prototypes/dau_v2_dashboard/dau_project && pip install -r requirements.txt && python server.py`
  - `cd prototypes/dau_v2_dashboard/dau_project && python generate_reference_plots.py`
  - `cd prototypes/dau_v2 && pip install -r requirements.txt && python -m dau_v2.self_check`

## Recommendations for follow-up

1. **Ruff / typing:** Run `ruff check prototypes --fix` and optionally add a minimal `ruff.toml` (line length, target version).
2. **Normalize legacy filenames** (e.g. `dau_active_inference(5).py`, `README (1).md`) when book links allow renames.
3. **DAU v3 static PNGs:** If those are also decorative AI mocks, apply the same pattern as v2 (delete + matplotlib or canvas-only).
4. **CI:** Add a lightweight workflow: `ruff check` + `python -m dau_v2.self_check` on PRs.
5. **Large binaries:** Consider moving historical result PNGs to releases or Git LFS if clone size becomes an issue.

## Confirmation — DAU v2 dashboard AI visuals

The only removed AI-style dashboard assets in scope were **`dashboard_interface1.png`–`dashboard_interface4.png`**. Live visualization remains **code-driven** (browser canvas + Python simulation). Optional static figures are produced by **`generate_reference_plots.py`** (matplotlib).
