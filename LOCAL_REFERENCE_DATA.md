# Local reference structures that cannot be redistributed

## MIL-101 — CCDC 605510

The locally validated MIL-101 file identifies:

- CCDC deposition: **605510**
- CSD structure DOI: <https://doi.org/10.5517/ccnb2lm>
- cited publications: <https://doi.org/10.1021/cm051870o> and
  <https://doi.org/10.1126/science.1116275>
- expected local filename: `605510.cif`
- verified local SHA-256:
  `3fcf3d682593c22ad7cfe9c2903f1f0b1b3a98c5a930a68738542724f6b937f6`

The file header states that it was downloaded from the Cambridge Structural
Database. CCDC's redistribution guidance does not permit external sharing of
original CSD data under an ordinary licence. Consequently, the CIF is not
stored in Git and is not included in the Zenodo/GitHub release archives.

Retrieve the file directly from the
[CCDC Access Structures service](https://www.ccdc.cam.ac.uk/structures/) using
deposition number `605510` or DOI `10.5517/ccnb2lm`, and accept the applicable
CCDC terms. Place the downloaded file locally at:

```text
local_reference_data/605510.cif
```

That directory is ignored by Git. Validate a downloaded copy without exposing
its atomic coordinates:

```powershell
python -m tools.validate_local_cif .\local_reference_data\605510.cif `
  --expected-sha256 3fcf3d682593c22ad7cfe9c2903f1f0b1b3a98c5a930a68738542724f6b937f6
```

The checksum identifies the exact local file used during release preparation.
CCDC may provide a different generated CIF in the future; a checksum mismatch
must be investigated rather than silently ignored.
