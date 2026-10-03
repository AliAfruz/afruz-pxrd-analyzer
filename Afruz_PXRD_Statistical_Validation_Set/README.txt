AFRUZ PXRD STATISTICAL VALIDATION SET
======================================

Primary test
------------
1. Import: 01_NaCl_raw_counts_with_sigma.txt
2. Load: NaCl_validation_P1.cif
3. Radiation: Cu Kalpha1
4. Wavelength: 1.5406 A
5. Use the full 4.0-80.0 deg 2theta range.
6. Select supplied sigma/uncertainty weighting when available.
7. Refine background, scale, zero shift, cell, and profile parameters gradually.

Normalization invariance test
-----------------------------
Repeat with 02_NaCl_normalized_100_with_sigma.txt.

Because this file is a pure linear normalization and sigma was scaled by the
same factor, Rwp, Rexp, GoF, and reduced chi-square should remain effectively
the same as the raw-count case. Small differences may occur from optimizer
convergence, but Rexp must not jump to values such as 100%.

Reference statistics against the exact generating profile
----------------------------------------------------------
Single-phase raw:
    Rwp  = 6.06433 %
    Rexp = 5.96986 %
    GoF  = 1.01582
    chi2 = 1.03190

Single-phase normalized:
    Rwp  = 6.06433 %
    Rexp = 5.96986 %
    GoF  = 1.01582
    chi2 = 1.03190

Two-phase raw:
    Rwp  = 5.91181 %
    Rexp = 5.93822 %
    GoF  = 0.99555
    chi2 = 0.99113

These exact reference values compare the noisy observations with the generating
numerical profile. A CIF-based refinement may differ because of profile,
scattering-factor, and optimization-model differences.

Mixed-phase mismatch test
-------------------------
Use 05_NaCl_CsCl_mixed_with_artifacts.txt with both CIF files.
Deliberate narrow artifacts are centered at:
    17.86, 46.22, and 71.34 deg 2theta.

They may be reviewed and unchecked for Pass 2. Do not mask genuine unexplained
Bragg peaks or broad residual regions merely to lower Rwp.

Scientific status
-----------------
This is deterministic synthetic software-validation data, not certified
experimental diffraction data and not a quantitative phase-analysis standard.
