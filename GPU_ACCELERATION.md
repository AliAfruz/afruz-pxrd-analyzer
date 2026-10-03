# Optional GPU acceleration

Afruz PXRD uses a hybrid calculation model so that enabling a GPU does not
change the scientific equations or make the application dependent on NVIDIA
hardware.

## Accelerated work

- CuPy/CUDA evaluates the expensive atom-by-reflection structure-factor matrix
  for large CIF powder-pattern synthesis and unfrozen Rietveld structure
  factors.
- Pawley and Le Bail use CUDA to build large point-by-reflection profile
  matrices. Very large Le Bail jobs also run the iterative profile-repartition
  matrix products on CUDA.
- PyQtGraph/OpenGL provides a rotatable, zoomable Crystal Studio view. Atoms
  are split into driver-safe shaded 3D sphere batches, with bonds, coordination
  polyhedra, cell edges and pore-volume envelopes sent to the GPU. Scene
  geometry is emitted as soon as it is ready. Pore scenes enforce a safe
  minimum camera distance so wheel zoom cannot enter and clip through the
  pore sphere.
- The GPU viewer can export its current camera to high-resolution PNG or TIFF
  through a tiled OpenGL framebuffer. Tiling permits output larger than one
  GPU texture; a reusable `.crystal.json` scene is saved beside the figure.

## Work that remains on CPU

- SciPy's nonlinear least-squares optimizer, Pawley non-negative bounded
  intensity solve, Le Bail bounded background solve, CIF parsing, geometry
  validation and bond/contact inference. These constrained solvers stay on CPU
  to preserve their validated behavior.
- The original Crystal Studio PNG/TIFF export stays on CPU as the deterministic
  publication reference. GPU export is intentionally a second, faster option:
  its antialiasing and shading can vary slightly with the graphics driver.

## Installation with an existing CUDA 13 toolkit

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Afruz discovers `CUDA_PATH` and the newest standard CUDA toolkit directory. Its
compiled-kernel cache is stored under `%LOCALAPPDATA%\AfruzPXRD\CuPyCache`.
If CUDA cannot start or a GPU calculation fails, the calculation is repeated on
CPU and the reason is recorded in the Rietveld Diagnostics tab.

## Using it

1. In Phase 11, leave **Use NVIDIA CUDA acceleration** checked and run the
   Rietveld refinement. Diagnostics reports the calculation backend, GPU and CPU
   optimizer separately.
2. In Phase 10, leave the same CUDA option checked for Pawley or Le Bail.
   Small problems remain on CPU automatically when transferring them to the GPU
   would be slower; Diagnostics reports which parts actually used CUDA.
3. In Crystal Studio, wait until **Open GPU 3D + fast export…** becomes enabled.
   Open it for smooth orbit/zoom inspection and use **Export GPU PNG / TIFF…**
   to save the live camera. Large CIFs automatically enable **GPU-first
   preview**, avoiding the slower CPU preview; uncheck it whenever you also
   want the deterministic preview. The normal Crystal Studio Export button
   always uses the CPU publication renderer and remains available.

The GPU window offers **Journal white** and **Dark studio** backgrounds.
Scientific and MOF scenes default to Journal white: it uses a pure white
framebuffer plus darker bonds, hydrogen bonds, polyhedron outlines and cell
edges for journal figures. The chosen background is preserved by GPU export
and recorded in the companion scene metadata.

## Atom transparency and a single-pore figure

- **Atom opacity** controls the atoms in both the deterministic CPU renderer
  and the interactive GPU renderer. Bonds, polyhedra and the yellow pore
  envelope keep their independent display opacity.
- Enable **Isolate one pore / cage** to retain one calculated maximal-empty-
  sphere envelope and a user-selected radial framework shell. Choose the pore
  number and increase **Framework shell** when linker or coordination-unit
  fragments are cut too closely.
- Leave **Smart framework-only cage** enabled for MOFs. It identifies bonded
  framework components first, estimates the void without disconnected guests,
  solvent or disorder sites, and retains only metal-connected cage-wall
  fragments within the closure shell. Disable it only for unusual metal-free
  or deliberately molecular structures whose connectivity cannot be inferred.
- The MOF presets retain plausible C–C linker contacts while independently
  suppressing inferred metal–metal contacts. Chromium uses a conventional
  lavender-blue scientific color, carbon dark gray, oxygen red and hydrogen
  white. The gold pore sphere defaults to high opacity so rear framework does
  not masquerade as atoms inside the void; lower **Yellow volume opacity**
  when deliberate see-through inspection is needed.
- Single-pore mode is a visualization crop of the periodic CIF, not a new
  molecular structure, adsorption calculation or proof of an independently
  stable cage. The exported scene records the selected pore and shell settings
  and carries this warning.

## Why CIF parsing and chemical topology remain on CPU

CIF tokenization, symmetry expansion, validation, bond/contact inference and
pore-search setup are branch-heavy operations. They were profiled on the
16,120-site MIL-101 model: scene construction took about 2.3 seconds for atoms,
4.0 seconds with contacts and 4.9 seconds with polyhedra. Moving those stages to
CUDA would add transfers and duplicate scientific code without a dependable
speed benefit. The expensive raster stage (about 15 seconds for the same scene
in the deterministic renderer) now has the GPU-first and tiled-export paths.
