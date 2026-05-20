# `mpas_grid_tstep0.h5m` Reference Guide

This is a parser-focused reference for `data/mpas_grid_tstep0.h5m`.
It mirrors the style of the `mpaso_roms_export_02.h5m` guide and adds a deep dive for the 3D ocean fields.

## 0) Quick Start For New Agent

If you only have a short context window, read these sections first:

- `10.1` Milestone and scope
- `10.4` Vertical extrusion convention (`ssh` + `layerThickness`)
- `10.5` Source chunking and halo rule
- `10.6` Current code status and file paths
- `10.7` Open risks / next checks

Current implementation location (as of 2026-02-19):

- `mfa-remap/src/cblock2.hpp`
- `mfa-remap/src/remap.cpp`

Key facts to preserve while continuing this work:

- bbox for source chunking is in MPAS XYZ (Earth-centered Cartesian)
- selected source elements = bbox-intersecting AABB set + one halo ring
- only active levels (`maxLevelCell`) are copied into `mpas_input`
- no sentinel values should be copied into MFA input
- per-level axial centroids use:
  - `top[k] = ssh - sum(thk[0..k-1])`
  - `bottom[k] = top[k] - thk[k]`
  - `centroid[k] = 0.5 * (top[k] + bottom[k])`
- downward convention: sea surface is highest axial value, depth decreases with level

Useful tag names (Polygon7):

- `maxLevelCell`, `timeDaily_avg_ssh`, `timeDaily_avg_layerThickness_3d`
- `timeDaily_avg_activeTracers_temperature_3d`
- `timeDaily_avg_activeTracers_salinity_3d`
- `timeDaily_avg_velocityMeridional_3d`
- `timeDaily_avg_velocityZonal_3d`

## 1) File Identity and Global Facts

- Format: MOAB H5M (HDF5 container)
- Source file: `data/mpas_grid_tstep0.h5m`
- File size: `6,742,473,116` bytes
- MOAB history: `MOAB`, version `5.5.1`, timestamp `04/30/25 07:31:26`
- HDF5 object stats:
  - groups: `36`
  - datasets: `27`
  - named datatypes: `27`
- Storage/layout:
  - all datasets contiguous
  - no chunking, no compression/filtering

## 2) High-Level MOAB/HDF5 Structure

Top-level tree under `/tstt`:

```text
/tstt
  /history                                    (4 strings)
  /elemtypes                                  (enum for MOAB element kinds)

  /nodes
    /coordinates                              (3980565, 3) float64
    /tags/GLOBAL_ID                           (3980565) int32

  /elements
    /Polygon7
      /connectivity                           (1971153, 7) uint64
      /tags                                   (20 datasets)

  /sets
    /list                                     (1, 4) int64
    /contents                                 (2) uint64
    /tags/PARTITION_METHOD                    (1) int32

  /tags                                       (26 tag definitions)
```

Entity start IDs (`start_id` attrs):

- nodes: `1`
- Polygon7 elements: `3980566`
- sets: `5951719`
- `/tstt/max_id`: `5951719`

## 3) Core Geometry and ID Conventions

- `nodes/coordinates` are Earth-centered Cartesian coordinates (ECEF-like), not unit sphere.
  - radius is effectively constant at `6371220` m (min/max around floating-point roundoff).
- `Polygon7/connectivity` stores 7 node references per cell.
- In this file, `GLOBAL_ID` is contiguous for both nodes and polygons (`1..N`), unlike some remapped exports.

## 4) Tag Inventory

Polygon7 has 20 attached tags:

- 5 vector (3D) tags: `timeDaily_avg_layerThickness_3d`, `timeDaily_avg_activeTracers_temperature_3d`, `timeDaily_avg_activeTracers_salinity_3d`, `timeDaily_avg_velocityMeridional_3d`, `timeDaily_avg_velocityZonal_3d`
- 15 scalar tags: `GLOBAL_ID`, `atmosphericPressure`, `bottomDepth`, `evaporationFlux`, `maxLevelCell`, `minLevelCell`, `rainFlux`, `salinityPistonVelocity`, `salinitySurfaceRestoringValue`, `seaIcePressure`, `temperaturePistonVelocity`, `temperatureSurfaceRestoringValue`, `timeDaily_avg_ssh`, `windStressMeridional`, `windStressZonal`

Tag definitions under `/tstt/tags` include mesh/set tags (`DIRICHLET_SET`, `MATERIAL_SET`, etc.) and `refBottomDepth` (global 80-level reference profile).

## 5) Vertical Model and Active-Level Rules

For Polygon7 cells:

- `nVertLevels = 80`
- `minLevelCell` is always `1`
- `maxLevelCell` ranges `2..80` (mean `62.514`, median `69`, p05 `16`, p95 `80`)
- active entries are levels `0..maxLevelCell-1`

Practical mask:

```python
active = (np.arange(80)[None, :] < maxLevelCell[:, None])
```

Depth distribution highlights:

- cells with `maxLevelCell <= 10`: `47,757`
- cells with `maxLevelCell >= 70`: `963,659`
- cells with full 80 levels: `146,723`
- active fraction of full `(nCells * 80)` storage: `0.781422`

## 6) Key 3D Fields (Deep Dive)

Equivalent key fields (relative to other MPAS exports):

- `layerThickness` -> `timeDaily_avg_layerThickness_3d`
- `temperature` -> `timeDaily_avg_activeTracers_temperature_3d`
- `salinity` -> `timeDaily_avg_activeTracers_salinity_3d`
- `velocityMeridional` -> `timeDaily_avg_velocityMeridional_3d`
- `velocityZonal` -> `timeDaily_avg_velocityZonal_3d`

All five are 80-level array-valued tags (`[80] float64`) per cell.

Important fill behavior in inactive levels:

- `timeDaily_avg_layerThickness_3d`: inactive entries are `0`
- `timeDaily_avg_velocityMeridional_3d`: inactive entries are `0`
- `timeDaily_avg_velocityZonal_3d`: inactive entries are `0`
- `timeDaily_avg_activeTracers_temperature_3d`: inactive entries are fill sentinel `-9.999999790214768e+33`
- `timeDaily_avg_activeTracers_salinity_3d`: inactive entries are fill sentinel `-9.999999790214768e+33`

Notes on statistics below:

- min/max/mean/std values are exact
- active-value percentiles (`p01/p05/p50/p95/p99`) are reservoir-sampled estimates (2,000,000 active values)
- surface and deepest-active percentiles are exact

### 6.1 `timeDaily_avg_layerThickness_3d`

- Active range: `0.200614 .. 146.032166`
- Active mean/std: `54.99994` / `52.07614`
- Active percentile estimates: p01 `1.98683`, p05 `2.57774`, p50 `31.61896`, p95 `143.35202`, p99 `145.36192`
- Surface stats: min `1.13306`, mean `1.98578`, p05 `1.98011`, p95 `1.98746`, max `2.46398`
- Deepest-active stats: min `0.200614`, mean `71.15974`, p95 `145.97472`, max `146.03217`

Column-consistency check:

- `sum(active layerThickness) - bottomDepth` has mean `-0.226762` and p95 abs `1.68252`
- this closely matches `timeDaily_avg_ssh` (difference from `sum(active layerThickness) - bottomDepth - ssh` has max abs `9.66e-05`)

### 6.2 `timeDaily_avg_activeTracers_temperature_3d`

- Active range: `-2.004314 .. 31.712069`
- Active mean/std: `7.69460` / `8.09836`
- Active percentile estimates: p01 `-1.42382`, p05 `-0.26285`, p50 `4.03961`, p95 `25.12475`, p99 `28.27443`
- Surface stats: mean `13.57970`, p05 `-1.37911`, p95 `28.36201`
- Deepest-active stats: mean `1.93611`, p05 `-0.78776`, p95 `9.48811`
- Active sign fractions: negative `6.59%`, positive `93.41%`

### 6.3 `timeDaily_avg_activeTracers_salinity_3d`

- Active range: `5.001590 .. 40.659237`
- Active mean/std: `34.65832` / `1.05290`
- Active percentile estimates: p01 `31.53297`, p05 `33.75283`, p50 `34.68104`, p95 `35.98877`, p99 `36.87235`
- Surface stats: mean `34.24404`, p05 `31.35509`, p95 `36.45971`
- Deepest-active stats: mean `34.55431`, p05 `33.69162`, p95 `34.96709`
- Active sign fractions: negative `0%`, positive `100%`

### 6.4 `timeDaily_avg_velocityMeridional_3d`

- Active range: `-2.203671 .. 1.998084`
- Active mean/std: `0.001716` / `0.105517`
- Active percentile estimates: p01 `-0.32204`, p05 `-0.12399`, p50 `0.000380`, p95 `0.12830`, p99 `0.33380`
- Surface stats: mean `0.011392`, p05 `-0.200176`, p95 `0.225410`
- Deepest-active stats: mean `0.000470`, p05 `-0.037082`, p95 `0.039407`

### 6.5 `timeDaily_avg_velocityZonal_3d`

- Active range: `-2.021015 .. 2.314289`
- Active mean/std: `0.014205` / `0.121066`
- Active percentile estimates: p01 `-0.29285`, p05 `-0.12184`, p50 `0.002099`, p95 `0.18986`, p99 `0.44894`
- Surface stats: mean `0.040822`, p05 `-0.250183`, p95 `0.350729`
- Deepest-active stats: mean `0.000843`, p05 `-0.035593`, p95 `0.040256`

### 6.6 Combined Horizontal Speed (`sqrt(u^2 + v^2)`)

- Surface speed: min `1.64e-05`, mean `0.18959`, p50 `0.13317`, p95 `0.52794`, p99 `0.97225`, max `2.42922`
- Deepest-active speed: min `3.45e-09`, mean `0.02385`, p50 `0.01088`, p95 `0.09230`, p99 `0.19046`, max `1.01874`

## 7) Scalar Field Highlights

Useful parser/context observations from scalar Polygon7 tags:

- `bottomDepth`: min `4.1541`, max `5549.7333`, mean `3438.4791`
- `timeDaily_avg_ssh`: min `-1.95927`, max `1.84643`, mean `-0.226762`
- `windStressZonal`: mean `0.05727`; `windStressMeridional`: mean `-0.00727`
- All-zero fields in this snapshot: `atmosphericPressure`, `evaporationFlux`, `rainFlux`, `seaIcePressure`
- Constant piston-velocity fields: `temperaturePistonVelocity = salinityPistonVelocity = 1.5856e-05`

## 8) Parser Checklist (Recommended)

1. Read `maxLevelCell` and build an active-level mask before touching any 3D field.
2. Treat inactive values carefully:
   - zero fill for thickness and velocity fields
   - large negative sentinel (`~ -1e34`) for temperature and salinity tracer fields
3. For tracer aggregates, never include inactive entries directly.
4. If reconstructing water-column thickness, use active levels only.
5. `GLOBAL_ID` is contiguous in this file, but keep parser logic robust to non-contiguous IDs in other exports.

## 9) Validation Basis

This guide is based on direct inspection with `h5ls`, `h5dump`, and `h5stat`, plus full-file streaming statistics computed with `h5py` for the five key 3D fields.

## 10) Session Handoff Notes (2026-02-19)

This section captures implementation decisions and code status from a live development session focused on MPAS source parsing for MFA remap in `mfa-remap`.

### 10.1 Milestone and Scope

Primary milestone for this session:

- update `CBlock2::readSourceData()` to accept a source read bounding box
- read only a chunk of MPAS source cells (plus halo)
- construct the `mpas_input` MFA `PointSet` correctly through geometry + source variable injection

Scope assumptions used:

- source variables are cell-centered (Polygon7 tags)
- vertical extrusion uses `maxLevelCell` active levels only
- sentinel values are never copied into MFA input
- bbox coordinate space is MPAS XYZ (Earth-centered Cartesian)

### 10.2 Key MPAS Tag Clarifications

Relevant Polygon7 tags used/confirmed:

- `maxLevelCell` (scalar, per cell)
- `timeDaily_avg_ssh` (scalar, per cell)
- `timeDaily_avg_layerThickness_3d` (array tag, length 80)
- 3D science vars (array tags, length 80):
  - `timeDaily_avg_activeTracers_temperature_3d`
  - `timeDaily_avg_activeTracers_salinity_3d`
  - `timeDaily_avg_velocityMeridional_3d`
  - `timeDaily_avg_velocityZonal_3d`

Important spelling reminder:

- thickness tag name is `timeDaily_avg_layerThickness_3d` (with `...Thickness...`, double `s`)

### 10.3 Observed `layerThickness` Examples (from `h5dump`)

Three sample cells inspected from `/tstt/elements/Polygon7/tags/timeDaily_avg_layerThickness_3d`:

- index `20` (`GLOBAL_ID=21`, `maxLevelCell=8`):
  - `[1.93361, 2.10932, 2.30107, 2.51031, 2.73859, 2.9877, 3.25958, 0.355618, 0, 0, ...]`
- index `25` (`GLOBAL_ID=26`, `maxLevelCell=44`):
  - starts near `1.98569`, increases through the column, then inactive tail is zero
- index `22490` (`GLOBAL_ID=22491`, `maxLevelCell=80`):
  - monotonically increasing for most of the column, with bottom value `49.6499` after very large deep values (`~145`)

Practical takeaway:

- inactive levels in `timeDaily_avg_layerThickness_3d` are zero-filled in this file
- active levels can be non-monotone at deepest levels in some cells

### 10.4 Adopted Vertical Extrusion Convention

Convention explicitly agreed for axial direction:

- sea surface is highest axial height value
- deeper levels are smaller height values (downward progression)

Per selected 2D source element `e` and active level `k`:

- `ssh = timeDaily_avg_ssh[e]`
- `thk[k] = timeDaily_avg_layerThickness_3d[e,k]`
- `top[k] = ssh - sum_{j=0..k-1}(thk[j])`
- `bottom[k] = top[k] - thk[k]`
- `axial_centroid[k] = 0.5 * (top[k] + bottom[k])`

XYZ embedding used in current code:

- take source polygon centroid in XYZ
- use normalized radial vector (centroid direction from origin) as local axial direction
- place level centroid at `xyz + axial_centroid * radial_unit`

### 10.5 Source Chunk Selection + Halo Rule

Selection policy implemented for source elements:

1. Build each Polygon7 element AABB from its 7 vertex coordinates.
2. Test AABB intersection against the provided source bbox (in bbox rotated space).
3. Mark intersecting elements as core selection.
4. Detect boundary core elements.
5. Add one halo ring: for each boundary core element, include all edge-neighbors.

Neighbor definition used:

- candidates are cells sharing at least one vertex
- retain neighbors sharing at least two vertices (edge-sharing polygons)

Expected property:

- bbox-intersecting elements are included
- an additional full 1-ring surrounds the intersecting set

### 10.6 Current Code Status (mfa-remap)

Implementation was made in:

- `/home/dlenz/spack/var/spack/environments/mfa-remap-env/mfa-remap/src/cblock2.hpp`
- `/home/dlenz/spack/var/spack/environments/mfa-remap-env/mfa-remap/src/remap.cpp`

Summary of implemented behavior:

- `CBlock2::readSourceData()` now takes `const mfa::Bbox<T>& sourceReadBbox`
- source subset is held in `selectedSourceElements`
- `computeSourceLevelInfo()` reads `maxLevelCell` only for selected elements
- `numSourceCells` is reset to `0` before rebuilding 3D indexing
- `addSourceVariable()` reads selected elements and fills only active levels
- executable was switched to `CBlock2` in `remap.cpp`

Current wiring in `remap.cpp` for bbox source selection:

- uses `sourceReadBbox(targetMins, targetMaxs)` as the initial integration path

### 10.7 Handoff Risks / Follow-up Suggestions

Items a new agent should verify next:

1. Validate geometric interpretation of radial axial direction for all target regions.
2. Consider CLI-driven bbox input (instead of deriving from target bounds only).
3. Benchmark halo neighbor build cost on full mesh; optimize if needed.
4. Confirm inactive handling for each variable tag (temperature/salinity sentinels should not be copied).
5. Add regression checks for:
   - selected cell count
   - max level consistency
   - no sentinel values in `mpas_input`
