# GitHub and Zenodo publication guide

This folder is a prepared source release. Complete the checks below before
making it public.

## 1. Confirm legal and identity metadata

- Confirm the recorded public identity: Ali Afruz, University of Mohaghegh
  Ardabili, ORCID `0000-0002-2969-8428`.
- Confirm ownership or redistribution permission for every source file and test
  fixture. Do not add unpublished experimental patterns or third-party CIFs
  without an appropriate license.
- Do not add the locally downloaded CCDC 605510 CIF to Git. Keep it in the
  ignored `local_reference_data/` directory and follow
  `LOCAL_REFERENCE_DATA.md`.
- Confirm that MIT is the intended software license. Changing a license after
  public contributions arrive can require contributor permission.

## 2. Run release verification

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1 -Development
.\run_tests.bat
python -m pip install build
python -m build
python -m pip install twine
python -m twine check .\dist\*
```

Do not publish if tests fail. Inspect warnings from the numerical-regression,
GUI, export, refinement, and Crystal Studio tests.

## 3. Create the GitHub repository

Create an empty public repository named `afruz-pxrd-analyzer`. Do not initialize
it with a second README or license. Then run from this release folder:

```powershell
git remote add origin https://github.com/AliAfruz/afruz-pxrd-analyzer.git
git push -u origin main
```

The repository URL is already recorded in `CITATION.cff`, `.zenodo.json`, and
`pyproject.toml`.

Recommended repository settings:

- enable Issues and private vulnerability reporting;
- protect `main` and require the Windows test workflow;
- disallow committed files larger than 100 MB;
- use squash or reviewed merge commits for scientific changes.

## 4. Connect GitHub to Zenodo

1. Sign in to Zenodo using the intended publishing account.
2. Open the Zenodo GitHub integration and enable the repository.
3. Verify that the title, author order, release version, license, and description
   in `.zenodo.json` are correct.
4. Create an annotated Git tag and push it:

```powershell
git tag -a v23.0.0 -m "Afruz PXRD Analyzer 23.0.0"
git push origin v23.0.0
```

5. On GitHub, create a release from tag `v23.0.0`. Attach the verified release
   ZIP and `SHA256SUMS.txt` produced with this package.
6. Zenodo should archive the GitHub release and mint a version-specific DOI.
   Review the Zenodo record before final publication.

## 5. Record the DOI

Zenodo provides both a version DOI and a concept DOI. Cite the version DOI for
exact reproducibility and use the concept DOI in general project documentation.
After the DOI exists:

- add it to `CITATION.cff`;
- add `doi` and `related_identifiers` to `.zenodo.json` as appropriate;
- add a Zenodo DOI badge and repository URL to `README.md`;
- cite the exact archived version in manuscripts.

Do not invent or pre-fill a DOI. Only Zenodo can assign it.

## 6. Suggested release description

> Afruz PXRD Analyzer 23.0.0 is a Windows-focused Python/PySide6 workbench for
> PXRD preprocessing, peak analysis, Pawley and Le Bail whole-pattern
> extraction, Rietveld refinement, validation, doping-series analysis,
> reproducible export, and crystallographic visualization. Synthetic validation
> fixtures are included for software testing; they are not certified reference
> materials. Scientific interpretation remains the responsibility of the user.
