AFRUZ PXRD ANALYZER — SYNTHETIC TEST SET
=============================================

Purpose
-------
This package is designed for testing import, peak finding, peak fitting,
multiphase Rietveld refinement, residual diagnosis, and Pass-2 mismatch masking.

Files
-----
NaCl_ideal_P1.cif
    Idealized, symmetry-expanded NaCl conventional cell written in P1.

CsCl_ideal_P1.cif
    Idealized CsCl primitive cell written in P1.

01_NaCl_single_clean.txt
    Clean single-phase NaCl test pattern.

02_NaCl_CsCl_mixed_clean.txt
    Clean two-phase NaCl + CsCl test pattern.

03_NaCl_CsCl_mixed_with_mismatches.txt
    Two-phase pattern containing deliberate narrow artifacts at:
    18.40, 42.76, and 67.18 degrees 2theta.

expected_reflections.txt
    Expected reflection positions and approximate simulated intensities.

Recommended mixed-phase test
----------------------------
1. Import 03_NaCl_CsCl_mixed_with_mismatches.txt.
2. Confirm the imported range remains 4.0–80.0 degrees 2theta.
3. Set radiation to Cu Kalpha1, wavelength 1.5406 A.
4. Add NaCl_ideal_P1.cif and CsCl_ideal_P1.cif as two required phases.
5. Run Pass 1 and inspect the residual/mismatch table.
6. The three sharp artificial spikes may be unchecked for Pass 2.
7. Do not hide the broad background hump near 23.5 degrees merely to reduce Rwp.
8. Run Pass 2 and compare the full difference curve and parameter stability.

Important scientific note
-------------------------
These are synthetic idealized files, not certified reference data and not
experimental measurements. They are suitable for software testing only.
Approximate phase scale amplitudes in the generated mixed pattern are not
certified weight fractions.
