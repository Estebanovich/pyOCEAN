# Writing guide for the pyOCEAN manual

This guide is for anyone (human or agent) who writes a chapter of `docs/manual/`.
It is not part of the published navigation.

## Audience and goal

A physical oceanographer who knows MITgcm, ROMS/CROCO, MOM6 or HYCOM, but has never seen pyOCEAN, must be
able to (1) understand exactly what the model solves and how, (2) set up and run a case, and (3) know how far
each feature has been validated. Be exhaustive: every option, every default, every equation the code uses.

## Ground rules

1. **The code is the truth.** Every equation, default value, option name and behaviour must be checked
   against the source. When `CLAUDE.md`, a README or a design doc disagrees with the code, follow the code
   and list the disagreement in your final report.
2. **No invented numbers.** Every quantitative result (errors, speeds, ratios, costs) comes from a README in
   `validation/`, a test in `tests/`, `CLAUDE.md`, `docs/design/*.md`, or a run you made yourself. Name the
   source next to the number (e.g. "(`validation/seamount/README.md`)").
3. **Cite the code as `path/to/file.py:LINE`** (in backticks; repo-relative paths). Cite the function or
   class name too, because line numbers drift: ``SigmaBaroclinic.step (`core/dynamics/sigma_baroclinic.py:2084`)``.
4. **Cite the literature** as Author (Year) in the text and give the full reference with DOI in a
   `## References` section at the end of the page. References you could not verify from the code docstrings,
   READMEs, `docs/literature/reading_list.md` or the local library are marked **[verify]**.
5. English, plain and direct. Short sentences. Explain the physics before the numerics, and the numerics
   before the code.
6. Do not edit any file outside `docs/manual/` and do not edit `mkdocs.yml`. Do not commit.

## Page template

```markdown
# Title

One-paragraph summary: what this part of the model does and when the user needs it.

## Physics / equations
## Discretisation
## In the code       (classes, functions, file:line, call order)
## Options and defaults   (table: option | values | default | meaning | where)
## Validation         (short: what was tested, result, link to the Validation page)
## Known limitations
## References
```

Adapt it as needed (user-guide pages are task-oriented: "how do I ...").

## Markdown features available (MkDocs Material)

- Math (MathJax via pymdownx.arithmatex): inline `$...$`, display `$$ ... $$` on their own lines with a blank
  line before and after. Inside tables, avoid `|` in math (use `\lvert`/`\rvert`). Symbols should match
  the code variable names where possible, and give units.
- Admonitions: `!!! note "Title"`, `!!! warning`, `!!! tip`, `!!! example`, collapsible `??? note`.
- Tabs: `=== "Tab name"`.
- Mermaid diagrams: a fenced block with language `mermaid` (flowchart TD / sequenceDiagram).
  Keep node labels short; quote labels that contain parentheses or symbols: `A["step (AB3)"]`.
- Tables: GitHub style.
- Images: `![Caption](../img/<chapter>/<name>.png)` (paths relative to the page). Add a one-line caption
  in italics below each figure saying what to look at.

## Figures

- Each chapter that draws figures has its own script `docs/manual/figures/make_<chapter>.py` that writes PNGs
  to `docs/manual/img/<chapter>/` (create the folder). Use matplotlib with the `Agg` backend, dpi 150,
  width ≤ 10 in, readable fonts (≥ 9 pt), colour-blind-safe colours, labelled axes with units.
- Schematics (C-grid stencil, layer geometry, ALE step, boundary zones, …) are encouraged: draw them with
  matplotlib patches/arrows, not ASCII.
- Figures from model runs: only small runs (< ~5 min on one core). Set `NUMBA_NUM_THREADS` ≤ 5 and the
  BLAS/OMP thread variables to 1. Never use more than 5 threads; never run two simulations at once.
  Run scripts with the base env: `/opt/anaconda3/bin/python`.
- Existing tracked figures in `validation/**/out/*.png` may be copied into `docs/manual/img/<chapter>/`
  (copy, do not move).
- Run your script and check that the PNGs exist (look at them with the Read tool).

## Final report (what you return)

- Files written (pages, scripts, images).
- Inconsistencies found between code and docs (CLAUDE.md, READMEs, docstrings) — with file:line.
- Anything you could not document or verify.

---

# Revision 2 (2026-10-14): one source per topic, one figure style, figures from model output

The first version repeated the same facts on many pages and every writer drew figures in a different style.
These rules override anything above that conflicts with them.

## One source per topic (text)

| Topic | The ONLY place it is written in full | Everywhere else |
|---|---|---|
| Validation numbers (errors, ratios, speeds, costs, scorecard) | `validation/*.md` (scorecard in `validation/index.md`) | at most one sentence without numbers + link to the exact validation section |
| Option names, defaults, valid values, error messages | `user-guide/configuration-reference.md` | prose says what an option does physically/numerically and links there; no option/default tables |
| What other models do (MOM6/CROCO/MITgcm/HYCOM features, migration tables) | `comparison.md` | keep only "ported from X `file:line`" where the code ports an algorithm |
| Complete list of limitations / open issues / planned work | `roadmap.md` | a page keeps ≤ 5 bullets specific to its topic, no numbers, link to roadmap |
| Module-by-module description of the code | `architecture/code-map.md` | — |
| Order of operations inside a step | `architecture/time-step.md` | theory pages explain each scheme and link to the step order |
| Model overview (what pyOCEAN is, model lines, capability table) | `index.md` (Home; `getting-started/overview.md` is merged into it and deleted) | — |

Inside a page: say each fact once. The opening summary is ≤ 4 sentences; no admonition that restates the body;
no "In short"/"Summary" sections that repeat the page. Keep each page's `## References` (only works cited
on that page). Each figure is embedded on one page only. Target: the manual 30–40 % shorter with no loss of
information — when you delete a duplicate, check the surviving copy is complete and correct, and turn the
deleted one into a link.

## One figure style

- Every figure script does `import pyocean_style as ps; ps.use()` (module in this folder; run
  `python pyocean_style.py` to see `img/style/style_sheet.png`) and saves with `ps.save(fig, chapter, name)`.
- Use `ps.figure(...)` for size (full width 7.5 in; never wider), `ps.model(key)` for every model/configuration
  curve (fixed colours across the manual), `ps.CMAPS[variable]` for every field (temperature → thermal,
  signed velocity/anomaly → balance centred on 0 with `TwoSlopeNorm` or symmetric limits, bathymetry → deep,
  amplitude → amp, phase → phase, error → matter), `ps.fill_bed` / `ps.BED` / `ps.WATER` / `ps.LAND` in
  sections and schematics, `ps.panel_labels(axs)` for multi-panel figures, `ps.colorbar` for colour bars.
- No figure titles (`suptitle`/long `set_title`): the Markdown caption explains the figure. Short panel
  headings (≤ 4 words) are fine. No source/explanatory sentences printed inside the figure.
- Schematics: few words, large labels (≥ 9.5 pt at full width), the physics drawn rather than written;
  arrows and symbols matching the equations. Prefer one clear schematic to three crowded ones.
- Data figures: units on every axis, legends without frames, no more than ~6 curves per panel, log axes when
  values span decades, reference/exact solutions in `ps.MODELS["exact"]` style.
- Captions in the Markdown: `*Figure: one or two sentences saying what to look at.*` under each image.
- Remove figures that duplicate another (e.g. three near-identical quick-start plots → one combined figure).

## Figures from model output (new)

Real model output is on disk; use it instead of re-running models. Read only what you need (some folders
are tens of GB); never run simulations except the tiny quick-start case. Never write into these folders.

| Output | Where | Contents (explore with `ls`, `np.load(..., allow_pickle=False)`, xarray) |
|---|---|---|
| Ilicak benchmark (lock exchange, overflow, internal wave, seamount; pyOCEAN z*/σ/hybrid, MOM6, CROCO, MITgcm, HYCOM) | `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/` | run folders, logs; readers in `validation/ilicak_benchmark/{series,mom6,croco,mitgcm}.py` |
| BayIW 2-D section, 3-D cut, IW realistic (pyOCEAN + MITgcm) | `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/{bayiw_2d,bayiw_cut,bayiw_iw,bayiw_2layer}/` | npz/nc fields and station series; readers in `validation/bayiw_*/compare.py` |
| MOM6 BayIW, MOM6 KE schemes, MOM6 lock exchange | `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/{mom6_bayiw,mom6_ke,mom6_lock_exchange}/` | MOM6 output; `validation/mom6_*/` readers |
| MITgcm internal_wave vs pyOCEAN | `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/{mitgcm,pyocean}` | `validation/internal_wave/compare.py` |
| PAC_MEX tides (26 runs) | `validation/pacmex_tides/out/run_*.npz` (+ `gauges_*.csv`) | harmonic constants per run; `compare_gauges.py`, `compare_wd.py`, `gulf_axis.py` |
| Lock exchange vs CROCO GRAV_ADJ | `validation/lock_exchange/out/lock_*.npz`, `~/pyOCEAN_refs/croco_runs/grav_adj/gravadj_his.nc` | |
| Thacker, TIDAL_FLAT 2DH/3D, BayIW barotropic seiche | `validation/{thacker,tidal_flat,bayiw_seiche}/out/*.npz`, CROCO runs in `~/pyOCEAN_refs/croco_runs/` | |
| Real bathymetry / data | `~/pyOCEAN_refs/` (GEBCO Alto Golfo, TPXO10 atlas, HYCOM GLBp0.08, tidal.rh, SAL) | |

From papers: never copy a published figure. You may plot analytical solutions and published *numbers*
(cited, with the table/figure they come from) next to pyOCEAN results.
