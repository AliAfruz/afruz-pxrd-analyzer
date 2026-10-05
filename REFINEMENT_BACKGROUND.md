# Shared refinement background engine — 23.4.0

This local update covers the regular Pawley/Le Bail and Rietveld workspaces,
their separate exclusion investigation windows, and Step-wise Refinement Studio.
The published GitHub/Zenodo release is unchanged.

## Choose a background

1. Select the experimental dataset. In Refinement Studio click **Background…**
   in the header, or **Background engine…** under Settings. In the regular
   refiners the same button is beside **Chosen background**, below Background order.
2. Select **Auto ensemble**, **arPLS**, **AsLS**, **SNIP**, **Rolling percentile**
   or **Polynomial**. This is the existing background engine, not a separate algorithm.
3. Adjust smoothness, asymmetry, iterations, window in degrees, percentile,
   polynomial order, peak protection, protection sigma and protection-width
   multiplier. Each method uses its applicable controls. **Clip negatives in
   preview only** affects the corrected preview, never refinement observations.
4. Click **Preview background**, inspect the raw/starting/corrected plots, quality
   diagnostics and ensemble-candidate table. Broad/amorphous scattering is not
   automatically distinguished from background. Review it scientifically.
5. Click **Use for all refinements**. The chosen curve is shared by the covered
   regular and step-wise methods for this dataset. Other datasets have their own
   selections. Refinement fits a correction on top, as requested.

If a background is already previewed in Pattern Preparation, **Use main
Background preview** imports that raw-unit curve and its settings. Alternatively
click **Use for all refinements** directly in the main Background section.
It does not require applying background subtraction first.

The plot's existing right-click copy/export tools are retained. **Add manual
peak** forwards to the main peak list when this same raw dataset is still
selected. **Export background TXT…** saves clean raw/starting/corrected-preview
columns and a separate parameters/diagnostics TXT file.

Parameter edits invalidate an uncommitted preview; preview again before choosing.
The editor refuses invalid/nonfinite/negative seed curves, a mismatched grid or
a changed raw dataset. Adjust the method/settings if the estimated baseline is negative.

## What refinement actually fits

The measured pattern is unchanged. The model is

`Icalc(2theta) = phase profiles + B0(2theta) + C(2theta)`.

`B0` is the saved chosen curve in raw measurement units. `C` is a Bernstein
polynomial whose degree is the refiner's **Background order** (separate from the
detection engine's Polynomial order). Its coefficients may be positive or negative
but are bounded below by `-min(B0)`. Since the basis functions are nonnegative
and sum to one, the total background stays nonnegative.

Internally, the existing bounded solver works on
`Iraw - (B0 - min(B0))`, with nonnegative background coefficients. This is a
reparameterization of the additive model, not preprocessing of the observations:
fit/reporting weights are computed from the original raw input, and the offset
is restored before residuals/statistics/plots are produced. The exported background
coefficients are the signed correction coefficients relative to `B0`.

Selecting a raw-unit background overrides **Use processed pattern** for covered
refinement runs, so smoothing, normalization, clipping or prior background
subtraction are not applied twice. The processed array itself is preserved.
GPU acceleration remains available for the existing accelerated profile/extraction
operations; the detection algorithms and bounded optimizer retain their CPU paths.

## Queues, projects and exports

- Choose the background before **Queue selected**. The raw input, seed curve,
  parameters and diagnostics are frozen into each attempt. Different exclusion
  masks slice measured and starting-background arrays consistently.
- Changing the choice affects fresh runs only. Existing queued/completed attempts
  retain their old choice. Discard and requeue pending attempts if you want a new one.
- Save **.AFZ** to retain the per-dataset choice, paused jobs and result curves.
  No background computation or refinement starts automatically on reopening.
- The XRD plot displays the chosen starting curve and fitted total background.
  Clean refinement profile TXT files include `chosen_background_intensity`,
  `fitted_background_correction` and the existing total `background_intensity`.
  Summary/provenance retains the method, parameters, diagnostics and checksums.
- **Restore ordinary fitted background** removes the shared choice for future
  runs, returning to the original jointly fitted polynomial background.

## Scientific limits

The detected starting curve is estimated from the same measurements, including
points later excluded from the refinement objective. Its statistical uncertainty
and effective degrees of freedom are not propagated. Consequently Rwp/Rp remain
fit measures, but Rexp, GoF and reduced chi-square are unavailable in this mode.
Intensity uncertainties are conditional/linearized diagnostics, not complete
uncertainties including background-selection uncertainty. A full-data background
also means held-out diagnostics are conditional, not independent validation.

Always retain a full-range control and report background choices, correction
order, excluded zones and reasons. A lower residual alone does not establish a
phase, impurity, atomic structure, abundance or publication-quality result.
