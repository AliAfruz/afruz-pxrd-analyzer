# Phase 12 — Doping-Series Comparison and Sequential Refinement

Release: 23.0.0

## Purpose

Phase 12 provides one auditable workspace for an ordered composition or doping series. It is designed to answer reproducible comparison questions: which peaks move, which features emerge or disappear, how the full pattern changes relative to a selected reference, and whether independently gated structure-constrained refinements show coherent trends.

## Workflow

1. Prepare each dataset in the main Background/Smoothing workspace when correction is scientifically justified. Phase 12 automatically checks **Prepared signal** when an applied prepared pattern exists and shows the recorded preparation steps in the series table.
2. Include at least two project datasets and record dopant, numeric concentration, and unit. Prepared signals affect comparison and peak search only; refinement always receives the untouched original intensity array and its raw-count provenance.
3. Use **Run linked Smart peak search** to run the same noise-aware Smart Peak Search used by the main Peaks workspace for every included pattern. Results are written to the main curated peak lists, preserve protected/manual peaks, respect frozen lists, and remain available to the rest of the project. Choose Conservative, Balanced, or Sensitive explicitly.
4. Select a reference composition and a transparent normalization rule: maximum, integrated area, a specified reference window, or raw intensity.
5. Compare the series with overlay, stacked, heatmap, and reference-difference views. The shared grid is restricted to the common angular interval and uses the coarsest measured median step. It is used only for visualization and differences.
6. Inspect matched peak tracks. Fitted/main curated peaks are preferred by default. If none exist, Phase 12 now falls back to the main Smart Peak Search instead of the older simple prominence detector; the source and count are recorded per profile. Position and d-spacing shifts are relative to the selected reference, and tracks absent from it are labelled emergent.
7. Optionally attach one or more CIF models and run either independent batch refinement or sequential refinement from the selected reference.
8. Review every pattern's optimizer status, Rwp, maximum parameter correlation, initialization source, warnings, and quality-gate decision before interpreting trends.
9. Export comparison tables, common-grid and difference profiles, refinement trends, observed/calculated/difference profiles, refined CIF copies, and a SHA-256 manifest.

## Sequential policy

The execution order is the selected reference, then increasing concentration, then decreasing concentration. An accepted neighbouring result may initialize the next cell, profile parameters, preferred-orientation value, and zero shift. A failed or rejected fit is never propagated. Each dataset is still refined on its own original measured grid and retains an independent quality gate. Independent mode always starts every dataset from the original CIF and settings.

This is sequential carry-forward, not a global parametric refinement with cross-dataset equations or constraints. Parametric Rietveld refinement can be valuable when a scientifically justified evolving model is available, but Phase 12 deliberately keeps that future capability separate from the present auditable screening workflow.

## Refined CIF boundary

Exported CIF copies update refined unit-cell scalars and append software, dataset, global displacement-factor delta, and scale-fraction audit fields. Input atomic coordinates and occupancies are retained. These files describe the fitted model; they do not demonstrate dopant location, site occupancy, oxidation state, phase purity, or equilibrium solid solubility.

## Interpretation safeguards

- A peak shift is not automatically lattice substitution. Verify wavelength, zero/displacement calibration, sample height, and uncertainty first.
- Intensity changes can arise from texture, absorption, thickness, preparation, preferred orientation, amorphous content, or phase-fraction changes.
- Overlap can make individual peak trajectories or phase parameters non-identifiable even when a fit looks visually good.
- Smooth trends are useful diagnostics, not independent validation. Inspect residuals, parameter correlations, uncertainty, phase assemblage, and external chemical evidence.
- The native Afruz Rietveld engine remains a screening/consistency engine and does not replace a validated FullProf, GSAS-II, TOPAS, or equivalent research refinement with an appropriate physical model.

## Verification references

- Stinton and Evans, *Parametric Rietveld refinement*, Journal of Applied Crystallography 40 (2007): https://journals.iucr.org/j/issues/2007/01/00/db5010/
- Roisnel and Rodriguez-Carvajal, *FullProfAPP for automated sequential refinements*, Journal of Applied Crystallography 57 (2024): https://journals.iucr.org/j/issues/2024/05/00/iu5063/
- Rowles, *pdCIFplotter*, Journal of Applied Crystallography 55 (2022): https://journals.iucr.org/j/issues/2022/03/00/yr5087/
- IUCr powder diffraction and pdCIF resources: https://journals.iucr.org/e/services/powder.html
