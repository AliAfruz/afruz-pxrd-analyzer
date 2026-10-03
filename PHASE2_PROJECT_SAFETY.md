# Phase 2 — Release and project hygiene

## Release

- Application: Afruz PXRD Analyzer
- Version: `22.1.1`
- Project schema written: `2`
- Project schemas readable: `1`, `2`

`afruz_pxrd.version` is the only source for the application name and release version. Window titles, the application header, About dialog, standalone exports, complete reports, validation packages, batch recipes, and project manifests derive their release identity from it. Format-specific schema versions remain independent.

## Atomic project-save contract

1. Serialize the project into a temporary file in the destination directory.
2. Validate the temporary ZIP CRC, required members, JSON manifest, supported schema, NPZ safety, and all dataset array alignments.
3. If the destination exists, copy it to a same-directory temporary backup and atomically publish that copy as `<project>.afz.bak`.
4. Atomically replace the destination with the already validated temporary project.
5. Remove temporary files on success or failure.

The existing project is not renamed or deleted before replacement. If the final replacement fails, the original project remains at its original path.

## Load and migration contract

Opening a project validates:

- Valid ZIP structure and CRC
- Exactly one `project.json` and `datasets.npz`
- UTF-8 and valid JSON
- Integer schema version in the supported set
- Dataset record identity and uniqueness
- Numeric, non-object, one-dimensional arrays
- Required raw/X arrays and aligned optional processed arrays
- Dataset-level monotonicity, finite values, and minimum length

Version 1 manifests receive defaults for optional UI, analysis, visibility, source-path, and metadata fields, then migrate in memory to version 2. A migrated project is only rewritten when the user saves it.

## Commands

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
.\run_windows.bat

powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1 -Development
.\run_tests.bat
```

## Verification result

Verified on 2026-08-11 with CPython 3.12.13 on Windows:

- `61 passed in 73.14s`
- Whole-package branch-aware coverage: `36.4%`
- `afruz_pxrd.project` branch-aware coverage: `74.9%`
- All 97 application/entry Python sources compiled and all package modules imported
- Version 1 migration and version 2 round-trip tests passed
- Invalid ZIP, missing-member, invalid-JSON, unsupported-version, and missing-array tests passed
- Simulated failure of the final atomic replacement left the original project byte-for-byte unchanged
- Second save created a readable `.afz.bak` containing the previous project state
- GUI title/header, TXT, Excel, complete JSON/Markdown/HTML reports, batch HTML, validation defaults, recipes, and project manifests reported `22.1.1`
- Launcher startup diagnostics passed and `pip check` reported no broken requirements
- Phase 0 NaCl and NaCl/CsCl scientific regression values remained within their frozen tolerances

Coverage HTML and JUnit XML were generated below the ignored `test_artifacts/` directory.
