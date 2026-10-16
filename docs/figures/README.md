# DIFFRUX figure gallery

These figures were generated with the application's own deterministic renderer
and Le Bail engine. They are not screenshots of another program, AI-generated
structures, or evidence of validation against experimental ground truth.

| Figure | Input and interpretation |
| --- | --- |
| `diffrux-hero.png` | Typographic banner and a CIF-derived 2×2×2 idealized NaCl view. Display radii/colors are illustrative. |
| `nacl-crystal.png` | One conventional NaCl cell, including periodic boundary sites, Na–Cl contacts and the central Cl coordination octahedron. Idealized reference fixture, not a fitted atomic structure. |
| `nacl-polyhedra.png` | Ali Afruz's selected original NaCl visualization, preserved byte-for-byte from a saved 23.5.0 synthetic Rietveld result. The refined cell is displayed with fixed CIF fractional coordinates/occupancies and periodic coordination completion. It is not the same run as the Le Bail plot. No pore sphere is present. |
| `nacl-le-bail.png` / `.svg` | Le Bail fit to the bundled raw-count NaCl fixture with supplied sigma, a TCH profile, cell/zero/profile refinement, 12 extraction cycles and staged optimization. Full measured range is shown without hidden exclusions. Synthetic software demonstration only. |

The two `*.crystal.json` files can be reopened in Crystal Studio. They retain
the input atomic model, camera, display settings, geometry and
warnings. Their source paths are repository-relative. `nacl-le-bail-profile.txt`
contains the observed, calculated, background and difference arrays plotted in
the fit figure. `gallery-manifest.json` records software identity, numerical fit
diagnostics, settings, input hashes and PNG hashes.

## Regenerate

From the repository root, using the normal Python 3.12 application environment:

```powershell
.\.venv\Scripts\python.exe tools\generate_gallery.py
```

Or use `python tools/generate_gallery.py` in another installed application
environment. Figures are written to this directory; pass `--output PATH` to
choose a different destination. The selected `nacl-polyhedra.png` is copied from
the committed original, not regenerated or visually edited. Its portable scene
supports reopening and optional manual rendering in Crystal Studio. Other
figures are regenerated. Rendering does not modify the CIFs, refiners,
or the user's saved projects. Font rasterization may differ across operating
systems; the source geometry and scientific settings remain recorded.

All figure inputs are project-generated fixtures under the repository's MIT
License. No CCDC/CSD CIF or private research measurement is redistributed.
See [data provenance](../../DATA_PROVENANCE.md).
