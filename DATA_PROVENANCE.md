# Data and fixture provenance

## Project-generated synthetic validation sets

`Afruz_PXRD_Synthetic_Test_Set` and
`Afruz_PXRD_Statistical_Validation_Set` contain deterministic synthetic patterns,
idealized NaCl/CsCl structures, and expected numerical references created for
software verification. They are not experimental measurements or certified
reference materials. They are distributed with this repository under the MIT
License for reproducibility and testing.

## MOF test fixture

`tests/fixtures/synthetic_cr_organic_framework.cif` is a hypothetical,
project-generated P1 framework used to test periodic MOF recognition, pore-scene
generation, and finite GPU/rendering inputs. It is not an experimental structure
or a model of a named compound. It is distributed under the repository's MIT
License.

## CCDC 605510 local validation reference

MIL-101 reference file `605510.cif` was downloaded from the Cambridge
Structural Database and identifies CCDC 605510, structure DOI
`10.5517/ccnb2lm`. Its local SHA-256 checksum is
`3fcf3d682593c22ad7cfe9c2903f1f0b1b3a98c5a930a68738542724f6b937f6`.

The original CSD CIF is **not included** in this repository or its release
archives. CCDC states that original CSD data must not be externally
redistributed under an ordinary CSD licence. Users must retrieve the structure
directly from CCDC and accept the applicable access terms. See
`LOCAL_REFERENCE_DATA.md`.

## User research data

No private experimental PXRD data, local refinement result, unpublished project
file, or user-specific Crystal Studio export is included in this release
candidate.
