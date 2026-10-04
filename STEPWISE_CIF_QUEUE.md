# Step-wise CIF candidate comparison

Local version 23.3.0. No GitHub or Zenodo update is included in this build.

## Open and run Step 1

1. Select your experimental XRD dataset. Open **Refinement & Quantification →
   Phase-exclusion investigation**, then its sequential refinement laboratory.
2. Choose **Add CIF(s)…** to import several models. Open the lower **Step-wise
   queue** tab; the imported CIFs appear in Step 1. The upper Active CIF selector
   is for the original manual run buttons, not for choosing queued candidates.
3. Check the **Run** boxes for the candidates you want. Edit their labels by
   double-clicking the Candidate label cells. Use Move candidate up/down to
   choose order, and optionally name the step.
4. Select the queue method: Pawley decomposition, Le Bail extraction, or
   Rietveld refinement. Set the fit range and other parameters under
   **Refinement controls**; choose **Use GPU where supported** if wanted.
5. In **Next-run exclusions**, label any regions to omit. Leave **Same range
   and exclusions for all candidates** checked for a matched-domain comparison.
6. Click **Queue checked candidates**, then **Start / resume this step**. Only
   one numerical worker runs at a time. Scroll within the queue tab if its
   lower action buttons are outside the visible area.

Each attempt receives a project-unique ID such as `AFR-RT-00003` and an ordered
label such as `S01 · C02 · RT · Candidate B`. PW, LB and RT in the history Type
column mean Pawley, Le Bail and Rietveld. Hover over a history cell for its full
label, CIF and parent ID; right-click to copy table data.

## Select results for Steps 2, 3 and later

1. Click history rows to review their frozen observed/calculated/difference
   plots. Check **Next** beside each completed result you wish to retain.
   Failed, cancelled, interrupted and explicitly non-converged results cannot
   seed a next step.
2. Click **Next step from checked results**. A new step contains only those
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
inherit different parent-specific ranges/masks instead, uncheck **Same range
and exclusions for all candidates** before queuing. Other current controls are
shared. A new domain signature records the actual observation, range, weighting
and numeric masks; matching signatures are necessary, not sufficient, for
meaningful R-factor comparison. Also review model complexity, profile/background
assumptions, residuals and chemical plausibility.

## Pause, change plans and save

- **Pause after current run** lets the active candidate finish and retains the
  remaining frozen jobs. **Cancel current run** requests cancellation at a safe
  numerical checkpoint and also stops further dispatch.
- **Continue queue after a failed candidate** controls whether subsequent
  pending jobs start after an exception. Failures remain in the audit history.
- Queuing freezes the measurement, CIF model, parameters and exclusions. Later
  draft edits do not alter pending jobs. To change them, first pause, use
  **Discard unstarted runs in this step**, edit, and queue fresh attempts.
  Discarded attempts are retained for audit, not deleted.
- Save the normal **.AFZ project** to retain CIFs, steps, labels, order, selections,
  settings, masks, results and frozen pending jobs. Large repeated atom/input
  snapshots are shared within the archive to reduce unnecessary duplication.
- Reopening never launches a job automatically. Select the desired step and
  press **Start / resume this step** to execute its pending jobs with their
  original IDs. A numerical run saved while active reopens as an interrupted
  snapshot; queue a fresh attempt for that candidate to obtain a new ID.
- Existing 23.2.0 histories remain readable. A legacy result lacking a model
  snapshot requires its CIF to be reimported before it can seed a new step.

## Export and scientific limitations

**Full history export (.zip)** includes the step candidate plan, history and
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
