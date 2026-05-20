# `mpaso_roms_export_02.h5m` Reference Guide

This document is a parser-focused reference for `data/mpaso_roms_export_02.h5m`.
It summarizes file layout, key MOAB conventions, and detailed statistics for the primary 3D ocean fields.

## 1) File Identity and Global Facts

- Format: MOAB H5M (HDF5 container)
- Source file: `data/mpaso_roms_export_02.h5m`
- File size: `1,573,216,396` bytes
- MOAB history: `MOAB`, version `5.5.1`, timestamp `11/04/25 18:29:45`
- HDF5 object stats:
  - groups: `111`
  - datasets: `105`
  - named datatypes: `100`
- Storage/layout:
  - all datasets contiguous
  - no chunking, no compression/filtering

## 2) High-Level MOAB/HDF5 Structure

Top-level tree under `/tstt`:

```text
/tstt
  /history                          (4 strings)
  /elemtypes                        (enum for MOAB element kinds)

  /nodes
    /coordinates                    (942873, 3) float64
    /tags/GLOBAL_ID                 (942873) int32

  /elements
    /Edge2
      /connectivity                 (33269, 2) uint64
      /tags                         (empty)
    /Polygon6
      /connectivity                 (465044, 6) uint64
      /tags                         (96 datasets, cell-attached fields)

  /sets
    /list                           (32, 4) int64
    /contents                       (34462) uint64

  /tags                             (99 tag definitions: type/class/default/global)
```

Entity start IDs (`start_id` attrs):

- nodes: `1`
- Edge2 elements: `942874`
- Polygon6 elements: `976143`
- sets: `1441187`
- `/tstt/max_id`: `1441218`

## 3) Important Parsing Conventions

### 3.1 Connectivity and Entity IDs

- `nodes/coordinates` gives geometry.
- `elements/*/connectivity` stores node references per element.
- `GLOBAL_ID` tags exist for nodes and polygons but are not contiguous sequences; treat as external IDs, not array indices.

### 3.2 Tag Definitions vs Tag Data

- `/tstt/tags/<tagName>` defines tag metadata (type, class, default/global fill).
- Actual values are under entity tag groups, for this file mainly:
  - `/tstt/elements/Polygon6/tags/<tagName>`

### 3.3 Vertical Data Model

For Polygon6 cells:

- `minLevelCell` is always `1` in this file.
- `maxLevelCell` ranges `2..64` (mean `54.016`, median `59`).
- Active vertical entries are levels `0..maxLevelCell-1`.
- Inactive tail entries are zero-padded for key 3D fields.

Practical mask:

```python
active = (np.arange(64)[None, :] < maxLevelCell[:, None])
```

Depth distribution notes:

- cells with `maxLevelCell <= 10`: `22,743`
- cells with `maxLevelCell >= 60`: `203,969`
- cells with full 64 levels: `59,226`
- active fraction of all `(nCells * 64)` entries: `0.843996`

## 4) Key 3D Fields (Deep Dive)

The most important fields are array-valued tags of length 64 per cell:

- `/tstt/elements/Polygon6/tags/layerThickness`
- `/tstt/elements/Polygon6/tags/temperature`
- `/tstt/elements/Polygon6/tags/salinity`
- `/tstt/elements/Polygon6/tags/velocityMeridional`
- `/tstt/elements/Polygon6/tags/velocityZonal`

All five:

- are conceptually shape `(465044, 64)`
- are stored as HDF5 array datatype `[64] float64` per cell
- use zero padding in inactive levels (verified: inactive nonzero count = `0` for all five)

### 4.1 `layerThickness`

- Active range: `1.94476 .. 5500.93`
- Active mean: `919.6122`
- Active percentiles:
  - p01 `10.0006`
  - p05 `30.0149`
  - p50 `327.449`
  - p95 `3792.68`
  - p99 `5004.47`
- Surface (`k=0`) stats:
  - min `1.94476`, mean `10.0448`, p05 `9.9889`, p95 `10.0061`, max `194.086`
- Deepest-active stats:
  - min `9.69128`, mean `3676.5263`, p95 `5500.15`, max `5500.93`

Important note for parser authors:

- Values increase strongly with depth level index and can reach ~5500 in deep cells.
- Summing this field over levels does **not** match `bottomDepth` in this file, so do not assume it is a straightforward per-layer thickness sum without validating downstream semantics.

### 4.2 `temperature`

- Active range: `-2.06444 .. 31.1939`
- Active mean/std: `9.2984` / `7.9557`
- Active percentiles:
  - p01 `-1.42824`
  - p05 `0.28244`
  - p50 `7.47636`
  - p95 `25.1787`
  - p99 `28.3429`
- Surface stats:
  - mean `18.3133`, p05 `-0.9651`, p95 `28.9524`
- Deepest-active stats:
  - mean `2.0928`, p05 `-0.6116`, p95 `9.2189`
- Active sign fractions:
  - negative `3.96%`
  - positive `96.04%`

### 4.3 `salinity`

- Active range: `5.00531 .. 56.1535`
- Active mean/std: `34.8199` / `0.8043`
- Active percentiles:
  - p01 `32.9755`
  - p05 `33.9456`
  - p50 `34.6997`
  - p95 `36.155`
  - p99 `37.031`
- Surface stats:
  - mean `34.6336`, p05 `32.5372`, p95 `36.6094`
- Deepest-active stats:
  - mean `34.6745`, p05 `34.4324`, p95 `34.9671`
- Active sign fractions:
  - negative `0%`
  - positive `100%`

### 4.4 `velocityMeridional`

- Active range: `-1.209 .. 0.974595`
- Active mean/std: `0.0004877` / `0.0312465`
- Active percentiles:
  - p01 `-0.0783132`
  - p05 `-0.0279704`
  - p50 `0.000227625`
  - p95 `0.029037`
  - p99 `0.0845183`
- Surface stats:
  - mean `0.0058321`, p05 `-0.0734307`, p95 `0.0879958`
- Deepest-active stats:
  - mean `0.0005578`, p05 `-0.0119522`, p95 `0.0144448`

### 4.5 `velocityZonal`

- Active range: `-0.955219 .. 1.31471`
- Active mean/std: `0.0061554` / `0.0539570`
- Active percentiles:
  - p01 `-0.119009`
  - p05 `-0.0515571`
  - p50 `0.00100348`
  - p95 `0.0811155`
  - p99 `0.171564`
- Surface stats:
  - mean `-0.0059877`, p05 `-0.1707021`, p95 `0.1510855`
- Deepest-active stats:
  - mean `0.0009493`, p05 `-0.0119817`, p95 `0.0164634`

### 4.6 Combined Horizontal Speed (`sqrt(u^2 + v^2)`)

- Surface speed:
  - min `1.63e-05`, mean `0.09727`, p50 `0.07368`, p95 `0.25347`, p99 `0.49365`, max `1.34279`
- Deepest-active speed:
  - min `5.38e-08`, mean `0.00958`, p50 `0.00471`, p95 `0.03375`, p99 `0.08093`, max `0.54241`

## 5) Additional Useful Fields for Parsing and Context

Common scalar cell fields under `/tstt/elements/Polygon6/tags`:

- geometry/grid: `lat`, `lon`, `area`, `aream`, `bottomDepth`, `mask`, `fCell`
- vertical control: `minLevelCell`, `maxLevelCell`
- surface snapshots: `temperatureSurfaceValue`, `salinitySurfaceValue`, `avgZonalSurfaceVelocity`, `avgMeridionalSurfaceVelocity`, `ssh`
- forcing/coupler families: `Faxa_*`, `Foxx_*`, `Fioi_*`, `Fioo_*`, `Fogx_*`, `Sa_*`, `Sg_*`, `Si_*`, `So_*`

Tag definitions exist for `hgt` and `refBottomDepth` under `/tstt/tags`, but there are no corresponding Polygon6 data arrays in this file.

## 6) Parser Checklist (Recommended)

1. Read mesh counts from dataset shapes.
2. Read `minLevelCell` and `maxLevelCell` first.
3. For 64-level fields, construct active mask from `maxLevelCell`.
4. Ignore inactive tail entries (zero-padded here, but still mask explicitly).
5. Do not assume `GLOBAL_ID` maps to contiguous 0-based indices.
6. Do not assume `layerThickness` can be blindly summed to reproduce `bottomDepth` in this export.

## 7) Validation Basis

This guide is based on direct inspection with `h5ls`, `h5dump`, and `h5stat`, plus full-field numeric scans for the five key 3D fields.
