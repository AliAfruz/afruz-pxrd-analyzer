# Contributing

Thank you for helping improve Afruz PXRD Analyzer.

## Development environment

Use 64-bit CPython 3.12 on Windows. From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1 -Development
.\run_tests.bat
```

## Scientific changes

Changes to peak positions, integrated intensities, background models, profile
functions, structure factors, weighting, uncertainty propagation, or agreement
indices must include:

1. the mathematical definition and units;
2. a primary literature or standards reference;
3. deterministic tests with explicit tolerances;
4. a comparison against an independent reference calculation when feasible;
5. documentation of defaults, constraints, and known limitations.

Do not alter raw measurements in place. Preserve the distinction between raw,
processed, normalized, and simulated arrays. New automated interpretations must
be phrased as diagnostics unless independently validated.

## Code changes

- Create a focused branch and keep unrelated changes out of the pull request.
- Add or update tests for every behavior change.
- Run the complete test suite before requesting review.
- Update `CHANGELOG.md` for user-visible changes.
- Do not commit virtual environments, caches, generated reports, private data,
  credentials, or unpublished experimental datasets.

By contributing, you agree that your contribution may be distributed under the
MIT License included in this repository.
