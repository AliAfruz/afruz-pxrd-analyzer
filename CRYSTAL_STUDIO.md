# Crystal Studio

Create a 3D crystal figure from a saved Rietveld result, with shaded atoms,
soft inter-atom shadows, element-colored bonds, translucent coordination
polyhedra, unit-cell edges and crystallographic axes. Rendering runs offline
on the CPU in a background worker; no extra application or GPU is required.

## Use it

1. Open the Rietveld refinement workspace and run a refinement.
2. Click **3D Crystal Studio…**, beside **Export result**.
3. Select a phase. Drag the figure to rotate it; scroll over the figure to zoom.
4. Choose **Cinematic • midnight**, **Publication • ivory**, **Studio • slate**,
   **Scientific • polyhedra + H-bonds**, or **MOF • porous framework**.
5. Adjust repeat counts along a/b/c, sphere size, bond thickness, contact cutoff,
   polyhedron center and opacity. Labels, axes, cell edges and captions are optional.
6. Click **Run Crystal Chemistry QA…** to inspect the frozen structural model.
7. Click **Export PNG / TIFF…**. Choose 1,600–4,000 pixels wide and a DPI value
   (600 by default). Enable transparency to place the figure on a manuscript or slide.

The editor has a scrollable controls panel. Render requests are coalesced while
dragging; the preview updates to the latest camera position when rendering finishes.
Export uses a frozen copy of the chosen phase and settings, so changing controls
during export cannot change the image being saved.

Each image is accompanied by a **.crystal.json** file. Open this file with
**Open CIF / saved scene…** to recover the exact structure and display settings.
The same scene metadata is also embedded in the PNG or TIFF. Saving over an image
updates its companion scene file too. Keep the pair together for editing later.
Studio figure export also writes **.chemistry-qa.json** and **.chemistry-qa.txt**
companions beside the image. The JSON preserves all site-level measurements and
assumptions; the text file is a compact laboratory-readable record.

The MOF preset is optimized for structures such as MIL-101(Cr): one displayed
unit cell, complete periodic metal-node coordination, thin linker bonds,
coordination polyhedra, hidden hydrogen atoms and a less crowded camera. It allows
up to 6,000 displayed atoms for a single-cell MOF view; repeated cells keep the
standard 1,800-atom safety limit. Hiding hydrogen changes only the illustration,
not the stored or refined structure.

**Open CIF / saved scene…** can also preview a standalone CIF. Such a view is
labelled **CIF reference**, rather than a refinement result.

### Scientific polyhedral preset

This preset follows the conventional visual language of crystallography figures:
a clean white background, compact atom spheres, thin bonds, a black outer cell box,
a vertical element key, strong red/green/blue crystallographic axes, and translucent
coordination polyhedra. It switches to one displayed cell, hides internal repeat-cell
grid lines, enables polyhedron outlines and geometric hydrogen bonds, and selects
**All metals** when the structure contains metals. Every setting remains editable.

Select **All metals** to draw polyhedra around Fe, Ni, or other metal centers in the
same figure; each center element uses its own conventional color. Select one element
when you need a less crowded coordination view. **Internal repeat-cell grid** restores
the subdivision lines used by the cinematic views. **Complete periodic boundary
coordination** adds only the outside-cell images used by a displayed bond, hydrogen
bond or polyhedron, so metal sites at a cell face retain a complete coordination
shell without filling the view with unused copies. Inferred metal–metal contacts are
off by default and require the explicit **Show inferred metal–metal contacts** control.

**Dashed hydrogen bonds** draws H···N/O/F/S contacts as gray dashes when H···A is
1.50–2.60 Å, D···A is 2.40–3.40 Å and the D–H···A angle is at least 145°. Only the
best geometric acceptor is shown for each H. These are display assignments, not
refined bond orders.

Every opened CIF and refined-cell snapshot is run through the structure plausibility
check. A weak/invalid model receives an orange **STRUCTURE REVIEW REQUIRED** banner;
ordinary publication export is locked. **Expert override: export flagged structure**
preserves access to a review image while retaining the banner and validation metadata.
The Rietveld workspace applies the same score-65 gate before numerical refinement.
Its expert override permits diagnostic refinement but does not turn the model into a
publication-ready structure or move any atom.

The overlap check is topology-aware. It excludes direct bonds and 1–3 contacts
(two atoms sharing a bonded neighbor), and classifies contacts that meet the stated
D–H···A geometry as hydrogen bonds before counting true nonbonded van der Waals
overlaps. This prevents normal water H···H geometry and hydrogen bonds from being
reported as atomic collisions. Generator scoring now penalizes the remaining true
nonbonded overlaps. When an explicit P1 atom list has no chemical formula, Afruz
exports its occupied cell composition, calculated formula weight and `Z = 1` with a
site-derived provenance tag; no coordinate or occupancy is changed.

### Crystal Chemistry QA

**Run Crystal Chemistry QA…** creates a report from the selected phase's frozen
fractional coordinates and current refined cell. It does not modify the model. The
report includes:

- periodic metal–O coordination numbers and every contributing ligand/image;
- M–O minimum, mean and maximum distances, bond-length distortion index, and
  ideal-angle RMS deviations for four- and six-coordinate polyhedra;
- site-level Fe and Ni bond-valence sums for explicitly stated oxidation-state
  assumptions;
- distance-based water-like, hydroxyl-like and carbonate-like connectivity;
- formal charge balance per explicit occupied cell;
- a screen for repeated translations and small-integer lattice point operations;
- saved Rietveld fit and correlation evidence when the report comes from a result.

Fe–O and Ni–O bond-valence sums use paired `R0` and `B` values from Gagné and
Hawthorne, *Acta Crystallographica B* **71** (2015), 562–578,
[doi:10.1107/S2052520615016297](https://doi.org/10.1107/S2052520615016297).
The default formal assumptions are Fe³⁺ and Ni²⁺; the JSON records the exact values
used. A good bond-valence or charge result supports plausibility but does not determine
oxidation state. Mixed valence, disorder and protonation need independent evidence.

The symmetry result is a screening result rather than a space-group assignment. It
tests whether element/occupancy-labelled sites map under translations and lattice-
compatible small-integer operations at 0.12 Å tolerance. Confirm any candidate using
the diffraction data, systematic absences and a dedicated crystallographic symmetry
program.

## What the geometry represents

- New Rietveld results save a deep copy of the actual input atomic model in each
  phase's `structure_snapshot`. These snapshots persist with the existing result
  and project serialization. The viewer uses that snapshot and that phase's
  `refined_cell`; it does not use the currently loaded or edited CIF.
- The current engine keeps fractional coordinates and occupancies fixed. A figure
  from its output depicts the final unit cell with those fixed atomic sites.
  Global ΔBiso is recorded in provenance; this renderer does not draw displacement
  ellipsoids or treat sphere sizes as measured atomic radii.
- Older results without atomic snapshots require a new refinement before result
  visualization. A separately opened CIF remains an explicitly labelled reference.
- The existing CIF parser expands supplied symmetry operations. The renderer uses
  these expanded sites once, adds periodic boundary copies for the displayed cells,
  and uses a right-handed Cartesian basis for non-orthogonal cells. It does not
  invent additional symmetry from a space-group name. Fractional-to-Cartesian
  conventions follow the [IUCr crystallographic coordinate description](https://www.iucr.org/__data/iucr/cifdic_html/3/CIF_IMG/CAXIS.html).
- Display contacts are a **distance heuristic**, not refined bonds. The three original
  styles connect mutual nearest shells with a 16% tolerance and a 6 Å search limit.
  The scientific preset uses a 24% tolerance around tabulated covalent-radius sums,
  which separates short O–H/C–O contacts from longer H···O contacts and retains
  metal–ligand contacts. Set an explicit cutoff when you know the appropriate
  coordination distance. Disable **Unlike elements only** for elemental or
  same-element networks. Metal–metal contacts remain separately disabled in the
  scientific preset unless intentionally enabled.
- Polyhedra are convex hulls of connected neighbors around the chosen element. In
  the scientific preset, metal polyhedra use the first N/O/F/S/Cl coordination shell
  within 4 Å (up to 28% beyond the nearest metal–ligand distance).
  With periodic boundary completion enabled, primary centers at a face use explicit
  neighboring-cell ligands and only enclosed shells with 4–24 neighbors are shown.
- Zero-occupancy sites are omitted. Partial occupancies are retained in metadata;
  sphere volume does not represent occupancy. Coincident disordered sites can hide
  each other and are reported in the view status.

## Figure quality and limits

The renderer uses orthographic projection, analytic sphere/cylinder surfaces,
depth-tested opaque geometry, soft key-light sphere shadows and sorted translucent
polyhedron faces. Translucent intersecting faces use a display approximation, not
full volumetric ray tracing. Smaller renders are supersampled for smoother edges.
These are structural illustrations, not electron-density or probability maps.

PNG and losslessly compressed TIFF preserve transparency. Both save pixel
dimensions and DPI metadata. DPI controls physical print size; it does not add
pixels. For example, 4,000 pixels at 600 DPI spans about 169 mm. Select the resolution
and final label size appropriate to the journal's figure requirements; the style
name does not certify compliance with any journal.

Views are bounded to 1,800 displayed atoms, 12,000 contacts and 16 megapixels per
export. Reduce repeats or the cutoff if a view exceeds the geometry limit.
The standard desktop export options stay below the image limit.

## Python use

```python
from afruz_pxrd.crystal_scene import RenderSettings, build_scene, models_from_result
from afruz_pxrd.crystal_render import export_crystal

model = models_from_result(rietveld_result)[0]
scene = build_scene(model, RenderSettings(repeats=(2, 2, 2)))
export_crystal(scene, "crystal.png", width=3200, height=2400, dpi=600)
```

For a CIF reference, use `model_from_structure(load_cif(path))` instead.
Pillow is now explicitly listed as a dependency; it was already present in the
project's pinned environment through Matplotlib.

## Checks

`tests/test_crystal_studio.py` covers skewed cell metrics, periodic boundary sites,
saved atomic provenance, contact controls, coordination hulls, occupancy validation,
depth/alpha composition, PNG/TIFF dimensions and metadata, deterministic scene
reopening, metal-contact suppression, periodic shell completion, hydrogen-bond
selection, export locking and the live Qt preview. The refinement-gate tests verify
that weak structures stop before fitting unless the explicit expert override is set.
`tests/test_crystal_chemistry_qa.py` covers exact octahedral geometry and BVS,
connectivity assignments, translational and point-symmetry candidates, immutable
input coordinates, and JSON/text report companions.
