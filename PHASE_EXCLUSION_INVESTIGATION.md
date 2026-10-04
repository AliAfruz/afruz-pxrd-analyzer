# Phase-Exclusion Investigation and CIF PXRD Simulation

Release: 23.1.0 local development build

## Purpose

The Phase-Exclusion Investigation workspace is an additive scientific tool. It
does not replace or modify the established Pawley/Le Bail and Rietveld
workspaces. It opens three independent windows:

1. Exclusion-aware Pawley refinement.
2. Exclusion-aware Le Bail extraction.
3. Exclusion-aware Rietveld refinement.

Each window uses the same validated calculation engine and parameter controls
as its production counterpart. Before the engine receives the data, a copied
dataset is filtered using the explicitly listed closed 2theta intervals. Raw
and processed arrays in the project remain unchanged.

## Using excluded regions

1. Select the experimental dataset and open **Refinement & Quantification ->
   Phase-exclusion investigation**.
2. Open Pawley, Le Bail, or Rietveld. These are separate non-modal windows, so
   the original refinement tabs remain available for the full-range control
   fit.
3. Move the amber selector on the profile and choose **Exclude selected
   region**, or enter a peak center and half-width and choose **Exclude peak**.
4. Give each interval a scientific label, for example `sample-holder peak`,
   `suspected secondary phase`, or `detector artifact`.
5. Run the refinement. Red crosses identify omitted measured points and shaded
   red bands show all excluded intervals.
6. Export the result. The bundle contains the included observed/calculated
   profile and a separate excluded-region audit table.

Overlapping or touching intervals are merged deterministically. At least 50
measured points must remain. The result records original, included, and
excluded point counts and states that calculated structural reflections inside
an excluded region were not deleted from the model.

## Scientific interpretation

Compare the full-range result against the exclusion-aware result. A large,
localized improvement after excluding an unmodeled peak can support further
testing of a secondary-phase or experimental-artifact hypothesis. It is not
proof of phase identity or phase purity. A publishable phase assignment should
include an explicit candidate phase model and, where possible, complementary
chemical or spectroscopic evidence.

Do not exclude unexplained regions only to reduce R factors. In a manuscript or
supporting information, report:

- every excluded 2theta interval and reason;
- the full-range and exclusion-aware R factors;
- parameters that changed materially between the two fits;
- whether an explicit impurity phase was tested;
- limitations from overlap, preferred orientation, background, fluorescence,
  absorption, specimen displacement, and instrument broadening.

## Simulated PXRD from CIF

Open **Phase & Structure -> Simulated PXRD from CIF**. Use the active CIF or
import another structure, choose wavelength and 2theta limits, then calculate.
The central canvas shows crystallographic reflection sticks and a broadened
pseudo-Voigt preview. The preview width and mixing parameter are display
controls only; no measured profile is fitted. Export produces separate clean
TXT files for the simulated continuous preview and the calculated reflection
table.
