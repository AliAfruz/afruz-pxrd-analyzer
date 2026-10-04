# Step-wise CIF candidate comparison

Local version 23.3.1. No GitHub or Zenodo update is included in this build.

## Open and run Step 1

1. Select your experimental XRD dataset. Open **Refinement & Quantification →
   Sequential refinement & exclusions**, then click **Open Refinement Studio —
   separate window**. The top Run button now says **Open Refinement Studio**
   and also opens it without requiring a global Phase & Structure CIF first.
2. Choose **Add CIF(s)…** in the Studio header to import several models. The
   left **Steps** tab shows the Step 1 candidates. The **CIFs** tab contains a
   separate Manual-run CIF selector and the preserved single-run buttons.
3. Check the **Run** boxes for the candidates you want. Edit their labels by
   double-clicking the Candidate label cells. Use **Move up / Move down** to
   choose order, and optionally name the step.
4. Select the queue method: Pawley decomposition, Le Bail extraction, or
   Rietveld refinement. Set the fit range and other parameters under
   **Settings**; choose **Use GPU where supported** if wanted.
5. In **Zones**, label any regions to omit. Leave **Shared range and exclusions**
   checked under Steps for a matched-domain comparison.
6. Click **Queue selected**, then **Start / resume**. Only one numerical worker
   runs at a time. These four run/step buttons are pinned outside the scroller.

Each attempt receives a project-unique ID such as `AFR-RT-00003` and an ordered
label such as `S01 · C02 · RT · Candidate B`. PW, LB and RT in the label mean
Pawley, Le Bail and Rietveld. Hover over a history cell for its full
label, CIF and parent ID; right-click to copy table data.

## Select results for Steps 2, 3 and later

1. Click history rows to review their frozen observed/calculated/difference
   plots. Check **Next** beside each completed result you wish to retain.
   Failed, cancelled, interrupted and explicitly non-converged results cannot
   seed a next step.
2. Click **Next step** under Steps. A new step contains only those
   candidates, in original attempt order, with explicit parent run IDs.
3. Adjust or retain the labelled exclusion zones and refinement controls. Pick
   the method again if you want to change it between steps.
4. Queue and start the checked candidates. Repeat result selection for Step 3
   and subsequent steps. Select an earlier step in the step selector to review
   its retained draft controls and exclusions.

The new run initializes from its selected parent's valid refined cell and
available zero-shift/profile parameters. Atom positions and occupancies remain
the CIF model's fixed sites; this workflow does not add atomic-site refinement.
Every candidate still fits the selected experimental measurement, not another
candidate's residual or a phase-subtracted pattern.

Shared current exclusions and ranges are the default for every new step. To
inherit different parent-specific ranges/masks instead, uncheck **Shared range
and exclusions** before queuing. Other current controls are
shared. A new domain signature records the actual observation, range, weighting
and numeric masks; matching signatures are necessary, not sufficient, for
meaningful R-factor comparison. Also review model complexity, profile/background
assumptions, residuals and chemical plausibility.

## Pause, change plans and save

- **Pause queue** lets the active candidate finish and retains the
  remaining frozen jobs. **Cancel current run** requests cancellation at a safe
  numerical checkpoint and also stops further dispatch.
- **Continue after failed candidate** controls whether subsequent
  pending jobs start after an exception. Failures remain in the audit history.
- Queuing freezes the measurement, CIF model, parameters and exclusions. Later
  draft edits do not alter pending jobs. To change them, first pause, use
  **Discard pending attempts**, edit, and queue fresh attempts.
  Discarded attempts are retained for audit, not deleted.
- Use the Studio's **Save .AFZ** button or normal project saving to retain CIFs, steps, labels, order, selections,
  settings, masks, results and frozen pending jobs. Large repeated atom/input
  snapshots are shared within the archive to reduce unnecessary duplication.
- Reopening never launches a job automatically. Select the desired step and
  press **Start / resume** to execute its pending jobs with their
  original IDs. A numerical run saved while active reopens as an interrupted
  snapshot; queue a fresh attempt for that candidate to obtain a new ID.
- Existing 23.2.0 histories remain readable. A legacy result lacking a model
  snapshot requires its CIF to be reimported before it can seed a new step.
- Closing the Studio (or pressing Escape) hides its single persistent window;
  it does not discard results or cancel active/queued jobs. Reopen it to inspect
  progress. Pause/cancel explicitly before hiding if you do not want later
  candidates to run. Keep the main application open while calculations run.

## Layout

The separate, non-modal window opens maximized and supports normal resize,
minimize and restore. Its full-height XRD plot is central. Left tabs are
**Steps**, **CIFs**, **Settings** and **Zones**; right tabs are **History** and
**Reflections**. Settings and zones scroll independently, long candidate tables
scroll internally, and the queue actions remain visible. Drag the two vertical
dividers to adjust plot/side-panel widths; **Reset view** restores the balanced
widths and plot autorange. Panel widths and active tabs are retained in AFZ.

## Export and scientific limitations

**Export history (.zip)** includes the step candidate plan, history and
parent IDs, comparison-domain signatures, every attempted run's provenance and
exclusions, frozen input-pattern TXT and model JSON, and clean refinement TXT
bundles for completed runs. Original source CIFs are included when still
available. Freeze/snapshot JSON is an internal parsed model, not a newly refined
atomic CIF. An export is an audit/data bundle; use AFZ for application resume.

Closed excluded intervals contribute neither observed nor calculated profile
points to that run's objective and fit statistics. Reflection centers in the
interval are marked inactive for plotting/audit; the structural hkl factors are
not deleted, and physical profile tails outside a mask remain part of the model.

This is **one-CIF-per-run candidate comparison**, not a simultaneous multiphase
fit, quantitative abundance determination or automatic impurity extraction.
Selecting a low-Rwp candidate or removing a region does not establish phase
identity, purity or dopant occupancy. Report excluded zones and reasons, retain
a full-range control, and validate candidate structures independently.

## Verification

Regression tests exercise all three native methods with two serial CIFs,
non-mutating frozen inputs, shared and parent-specific masks, Steps 2/3 parent
selection, GPU-request snapshotting, failure/pause/cancel behavior, AFZ pending
resume, snapshot sharing, historical plotting and full-history export.
