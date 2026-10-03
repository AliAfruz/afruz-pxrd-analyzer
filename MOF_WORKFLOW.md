# MOF refinement and visualization

Afruz now includes conservative presets for large-cell porous frameworks such as
MIL-101(Cr), plus a dedicated **MOF • porous framework** style in Crystal Studio.

## Recommended first pass

1. Import the experimental PXRD pattern and its measurement metadata.
2. Import a chemically credible, activated or guest-containing CIF that matches
   the measured sample state.
3. In **Pawley / Le Bail**, click **MOF refinement preset**. The preset selects
   Le Bail extraction, retains weak reflections, and refines the cell and profile
   in stages. When the imported dataset extends beyond 35° 2θ, the preset chooses
   a ≤35° low-angle first pass; this is deliberate for the dense reflection set
   of large-cell frameworks. Use this first to test cell/profile agreement without
   relying on calculated atomic intensities.
4. In **Rietveld**, click **MOF refinement preset**. Atomic coordinates,
   occupancies, global Biso and preferred orientation remain fixed. The unit cell,
   scale, background, zero shift and profile are the controlled first-pass terms.
5. Open **3D Crystal Studio…** and choose **MOF • porous framework**. This uses a
   single cell, thin linker bonds, metal-node coordination polyhedra, complete
   periodic boundary coordination and hidden hydrogen atoms. All controls remain
   editable, and the saved structural model is unchanged.

## Scientific limits

MOF patterns often contain intense low-angle peaks, severe reflection overlap,
guest/solvent disorder, framework defects and broadening caused by small domains.
Do not interpret a lower Rwp alone as proof of pore occupancy, defect chemistry or
linker position. The current native Rietveld engine keeps fractional coordinates
fixed and uses approximate scattering factors; use GSAS-II, TOPAS, FullProf or
another validated crystallographic package for publication-grade coordinate,
occupancy and displacement-parameter refinement.

For MIL-101(Cr), verify the CIF state (hydrated, activated or guest-loaded),
preserve the lowest reliable 2θ region, use the measured wavelength/instrument
profile, and inspect residuals around the strongest low-angle cage reflections.
After the first fit is stable, extend the upper 2θ limit in controlled steps and
confirm that the reflection list and runtime remain suitable for the large cell.
