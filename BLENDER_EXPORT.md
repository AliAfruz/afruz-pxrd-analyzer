# Editable Blender crystal export — 23.5.0

The same export is available in **GPU Interactive Crystal** and the main
**Crystal Studio**. It uses the prepared frozen scene, not a screenshot, and
does not change CIF parsing, refinement, pore detection or rendering geometry.

## Export and import

1. Prepare the desired CIF/reference or saved refinement in Crystal Studio.
   Choose the supercell, atom transparency, polyhedra and optional single pore.
   Wait until the current geometry is ready.
2. In the GPU window click **Export editable 3D for Blender…** below the PNG/TIFF
   controls. The same button is in the main Studio's Export section.
3. Choose sphere mesh quality and included parts, then **Export editable 3D (.glb)…**.
   High quality uses 1,280 faces per shared sphere. The export runs in a worker
   with progress; it does not depend on DPI or a GPU framebuffer.
4. Blender: **File → Import → glTF 2.0 (.glb/.gltf)**. Choose the `.glb`.
   Select the imported mesh objects (**A** in a fresh scene), then **View → Frame Selected**. Use Material Preview
   to see colors and transparency. Add your own lights/camera for a graphical abstract.
5. Save **.blend** to retain Blender-specific editing.

Blender's [official import documentation](https://docs.blender.org/manual/en/5.1/addons/import_export/scene_gltf2.html)
describes the GLB importer. The exporter follows the
[glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).

## What is editable

- **Atoms**: separate named objects, with element, original site label, occupancy,
  displayed Cartesian/fractional coordinates and periodic-helper status as custom properties.
- **Bonds**: separate cylinders with atom-index connections and Cartesian distances.
  These represent the existing display contact graph, not new measured bond orders.
- **Polyhedra**: exact current triangular surfaces, grouped into editable meshes
  by center element. Optional edge guides are separate cylinders.
- **Pore envelopes**: separate smooth orange spheres with saved centers/radii.
- **Hydrogen bonds**: grouped dashed cylinders. **Unit cell**: separate edge guides.

Sphere/cylinder mesh data and element materials are shared to keep large MOFs
compact. Objects can be moved/rotated independently. Before changing just one
atom's mesh, use **Object → Relations → Make Single User → Object & Data**.
Copy its material too if its color/transparency must be independent. Otherwise
editing shared geometry/materials affects all objects using those data.

## Coordinate accuracy and limits

One Blender unit numerically represents **one Ångström** for convenient graphics.
The GLB's metre geometry is intentionally magnified by **10^10** relative to the
physical structure; it is not a life-size/SI manufacturing model. A rigid root
rotation converts Studio Z-up to glTF Y-up. Blender's importer restores the
Cartesian XYZ convention without mirroring the structure.

Atom centers and contact endpoints use the frozen displayed Cartesian coordinates.
Mesh vertex buffers use glTF's standard float32 precision; node transforms retain
the source numeric coordinates. Sphere tessellation approximates curved surfaces.
Single-pore scenes include the same uniform display translation/crop as Studio;
the companion source retains the original CIF/refinement model.

Atom radii, cylinder widths, coordination hulls and the orange pore sphere are
**visualization conventions**. A pore sphere is a grid-derived maximal-empty-sphere
estimate, not an atom, refined parameter, adsorption-accessible surface or complete
pore-size analysis. A cropped cage is not an independently extracted molecule.
Partial occupancies/disorder are metadata, not replaced by an invented ordered model.
Pawley/Le Bail views retain CIF atomic sites; a rendered fit is not proof of atomic refinement.

Lighting and transparency can differ from GPU screenshots. Bonds become cylinders
instead of screen-space lines. Text labels, captions, axes, image backgrounds and
the live GPU camera are not baked as mesh objects; camera values, when available,
are retained as provenance in the companion source.

## Source and safety

The bundle contains `.glb`, a reusable `.crystal.json` source and a short
`_blender_README.txt` guide. The source includes frozen structure, cell, refinement
provenance, settings, warnings and the GLB checksum. Nonfinite *diagnostics* are
recorded explicitly as strings; nonfinite geometry is rejected before files are written.

Structures marked as requiring review need an explicit expert override. Export
preserves that review flag; it never certifies the structure. Previously overwritten
files are kept as timestamped backups. Geometry export does not refine or optimize
atoms. Very large scenes are bounded; reduce repeats, hide hydrogen or isolate a
pore to keep Blender responsive.

This is a local update. No GitHub or Zenodo publication is included.
