# Phase-Exclusion Investigation and CIF PXRD Simulation

Release: 23.3.1 local development build

## Purpose

The Phase-Exclusion Investigation workspace is an additive scientific tool. It
does not replace or modify the established Pawley/Le Bail and Rietveld
workspaces. Its main view is now the **Sequential Multiphase Refinement and
Exclusion Laboratory**, while the three independent 23.1 windows remain
available:

1. Exclusion-aware Pawley refinement.
2. Exclusion-aware Le Bail extraction.
3. Exclusion-aware Rietveld refinement.

Each window uses the same validated calculation engine and parameter controls
as its production counterpart. Before the engine receives the data, a copied
dataset is filtered using the explicitly listed closed 2theta intervals. Raw
and processed arrays in the project remain unchanged.

## Central sequential-refinement laboratory

Open **Refinement & Quantification -> Sequential refinement & exclusions ->
Open Refinement Studio**. The separate window's large central plot shows either the selected experimental pattern or a
frozen historical refinement. Add any number of CIF files to the library and
select one active CIF for the next run. The three buttons under **CIFs** start Pawley,
Le Bail or Rietveld refinement directly from that CIF. Raw measured intensity
is the default input; processed intensity must be selected deliberately.

Every attempted refinement receives an immutable human-readable identifier,
for example `AFR-RT-00003`, plus a parent ID when it continues a selected run.
The history at the right records method, active CIF, status, Rwp, user label,
settings, timestamps, exclusions, GPU provenance and the complete numerical
result. Selecting a history row restores its frozen observed/calculated/
difference profile. **Continue from selected run** restores that run's CIF,
settings and exclusion set as the starting point for a new child run; the old
record is never overwritten.

To prepare the next refinement, open **Zones**, drag the amber selector or left-click the plot
to center it, provide a scientific label, and add a region or peak-width
exclusion. The next run omits both measured and calculated profile points in
those closed intervals from its objective and fit-domain statistics. CIF
reflection centers inside a mask are marked `fit_active = false`, hidden from
that run's active Bragg ticks, and exported with the label. The hkl structure
factors are not deleted: deleting selected crystallographic reflections from a
Rietveld structure model would be physically invalid and could suppress real
profile tails outside the interval.

**Export history (.zip)** creates one ZIP containing the lineage table, workspace
manifest, a clean full refinement TXT bundle for every completed ID, applied
exclusion tables, provenance, reflection activity and available source CIFs.
AFZ project saving stores the CIF library, run counter, complete history,
draft exclusions, selected history item and controls. Reopening the AFZ lets a
new child refinement continue from any completed record. A numerical optimizer
that was interrupted while running is recorded as interrupted and is restarted
as a new ID; its in-memory iteration cannot be resumed safely.

## Step-wise CIF candidate queue

The Studio's left **Steps** tab extends the same laboratory with automatic serial
candidate runs and result selection for follow-up steps. Each run fits just
one CIF, as opposed to fitting active CIFs together. See
[STEPWISE_CIF_QUEUE.md](STEPWISE_CIF_QUEUE.md) for the complete procedure,
pause/reopen behavior and export contents. Established single-run buttons and
the independent investigation windows remain available.

## Using excluded regions

1. Select the experimental dataset and open **Refinement & Quantification ->
   Sequential refinement & exclusions**.
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
measured points must remain for Pawley/Le Bail, and 80 for the laboratory's
Rietveld runs, inside the selected fit range. The result records original, included, and
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
