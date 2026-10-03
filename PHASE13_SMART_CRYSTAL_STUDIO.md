# Phase 13 — Universal Smart Crystal Studio

Phase 13 adds an opt-in, deterministic view planner for general CIF and saved
Rietveld structures. It does not change atomic coordinates, occupancies, unit
cells, phase fractions, profile parameters, or any refinement result.

## What the planner examines

- Occupied-site composition and partial occupancies.
- Element-aware periodic distance contacts using covalent radii ×1.24.
- Independent lattice translations in connectivity cycles (0D, 1D, 2D, 3D).
- CIF `_geom_bond_*` records when the author supplies them.
- Model size and topology for camera, repeat-cell, detail and GPU-first advice.

The report gives the assigned family, confidence, evidence, topology dimension,
connectivity source, recommended controls, and scientific limitations. Manual
controls remain authoritative after a plan is applied.

## Scientific boundaries

Automatic contacts are display/topology assignments, not refined bond orders.
The planner does not infer oxidation states, protonation, magnetic ordering, or
adsorption-accessible porosity. A pore classification is only a candidate; use
a validated probe-accessible surface/volume program before making porosity or
adsorption claims. Partial occupancies and unknown sites are reported rather
than silently repaired.

CIF geometric bond rows are preserved separately. The renderer matches their
site labels and reported distances and reports how many were used. Non-identity
symmetry codes are retained verbatim and are not presented as fully resolved
unless the displayed geometry independently matches.

## Compatibility and performance

The feature is opt-in through **Analyze + apply smart view**. Existing MIL-101
and UiO-66 presets, single-pore extraction and GPU rendering remain available
unchanged. Topology analysis is bounded at 6,000 occupied sites; larger models
receive a conservative single-cell/GPU-first view without an expensive topology
claim. Rendering and analysis remain background tasks so the Qt interface stays
responsive.
