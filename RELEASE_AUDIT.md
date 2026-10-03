# Release audit — version 23.0.0

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
