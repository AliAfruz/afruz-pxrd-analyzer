# Release audit

## DIFFRUX 23.6.0 — GitHub source update, 2026-10-16

- Application identity: DIFFRUX (formerly Afruz PXRD Analyzer).
- Interpreter: CPython 3.12.14; Qt/PySide6 6.11.1; Windows offscreen tests.
- Complete current suite: **352 passed in 523.20 s**, with no failures or skips.
- After replacing the gallery's orange-pore image with Ali Afruz's selected NaCl
  polyhedra image, all six gallery/branding tests were rerun: **6 passed in 0.66 s**.
- A real CUDA smoke test and the large Le Bail CPU/CUDA parity check ran on an
  NVIDIA GeForce RTX 4060 Laptop GPU. The parity test asserts actual GPU use when
  CUDA is available; CPU-only installations retain the tested fallback.
- Regular/step-wise refiners, shared backgrounds, exclusions, AFZ recovery,
  typed results, rendering, editable GLB export and numerical baselines are covered.
- NaCl Le Bail demonstration: Rwp approximately 6.11% with the recorded TCH
  settings and supplied uncertainties. This is a synthetic fixture result, not
  a promise about experimental data or independent scientific validation.
- Public figures use project-generated NaCl inputs. The selected original PNG
  SHA-256 is `2d14c7212457546defbeda21ff5ec994d0c14ec1c35cf1d247803a8a3a015eaa`.
  Its sidecar uses a portable source path and preserves the historical fitted cell.
- No pore-sphere figure, private measurement, user project, manuscript package,
  third-party CSD CIF, environment, cache or local backup is added to GitHub.
- Existing `.afz` files, scene schemas, settings/cache keys, import/distribution
  names and legacy launch commands are retained. The `diffrux` command is an alias.
- Updating GitHub does not change the historical Zenodo 23.0.0 archive or create
  a new DOI. Citation metadata distinguish the project concept DOI from that archive.

## Historical release audit — version 23.0.0

- Release candidate date: 2026-10-03
- Supported interpreter: CPython 3.12
- Test interpreter: CPython 3.12.14
- Qt/PySide6 test runtime: 6.11.1
- Source worktree base commit: `1987db674a37d83cf2329e8baf28e223790dcdf3`

## Verification result

The clean release snapshot was tested independently of the development folder:

```text
224 passed in 403.15 s
```

The test run used the offscreen Qt platform and Agg Matplotlib backend. It
covered scientific processing, statistical metrics, Pawley/Le Bail and Rietveld
regressions, central state, import/export, report generation, GUI construction,
GPU safety/fallback behavior, MOF workflows, Crystal Studio, and structure QA.

The release metadata passed JSON and TOML parsing. A universal Python wheel and
source distribution were built successfully from `pyproject.toml`.

The downloaded CCDC 605510 MIL-101 CIF was loaded locally as an external
reference and classified as a metal–organic framework. It is intentionally
excluded from every source and binary distribution because CCDC does not permit
ordinary redistribution of original CSD data.

## Deliberately excluded from the public snapshot

- the development virtual environment;
- Python, pytest, and coverage caches;
- temporary validation installations;
- local backup directories;
- generated screenshots and Crystal Studio scene exports;
- local MIL-101 refinement results;
- old release ZIP files and nested package archives;
- baseline artifacts containing local generated reports or project snapshots.
- downloaded CSD CIF files, including CCDC 605510.

## Limitations of this audit

Passing software tests does not certify scientific results. No claim is made
that synthetic fixtures are certified reference materials, that every CIF is
chemically correct, or that a refinement is publishable without expert review.
External reference-program comparisons and instrument-standard validation
remain necessary for the scope of each research study.
