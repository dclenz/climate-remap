# `roms_3d_projected` MOAB H5M Reference Guide

This document summarizes two ROMS-projected MOAB HDF5 meshes and is intended as a parser-oriented reference.

- `data/roms_3d_projected_BoB.h5m`
- `data/roms_3d_projected_niskine.h5m`

Note: the requested path `data/roms_3d_projected/niskine.h5m` does not exist in this workspace; the analyzed file is `data/roms_3d_projected_niskine.h5m`.

## 1) Shared MOAB/HDF5 Schema

Both files use the same high-level structure:

```text
/tstt
  /history
  /elemtypes

  /nodes
    /coordinates                       (Nnodes, 3) float64
    /tags/GLOBAL_ID                    (Nnodes) int32

  /elements
    /Quad4
      /connectivity                    (Ncells, 4) uint64
      /tags                            (19 datasets)

  /sets
    /list                              (3, 4) int64
    /contents                          (10) uint64

  /tags                                (definitions; includes some tags not attached to Quad4)
```

Key shared points:

- Element type is `Quad` (`Quad4` connectivity).
- Node coordinates lie on a unit sphere (radius ~1.0 in both files).
- `GLOBAL_ID` is contiguous for nodes and cells in both files.
- Quad4 has exactly 5 vector fields of length 100 and 14 scalar fields (19 total tags).

## 2) Per-File Core Dimensions and IDs

| Metric | `roms_3d_projected_BoB.h5m` | `roms_3d_projected_niskine.h5m` |
|---|---:|---:|
| File size (bytes) | 334,091,052 | 91,061,148 |
| Nodes | 80,949 | 22,165 |
| Quad4 cells | 80,364 | 21,868 |
| Connectivity width | 4 | 4 |
| `/tstt/max_id` | 161,316 | 44,036 |
| Node `start_id` | 1 | 1 |
| Cell `start_id` | 80,950 | 22,166 |
| Set `start_id` | 161,314 | 44,034 |

Coordinate ranges (x, y, z):

- BoB: x `[-0.09449, 0.16802]`, y `[0.91169, 0.98244]`, z `[0.16934, 0.41084]`
- niskine: x `[0.46050, 0.51399]`, y `[-0.23567, -0.17662]`, z `[0.83605, 0.86000]`

Interpretation: same unit-sphere representation, different geographic subdomains.

## 3) Quad4 Tag Inventory

Common Quad4 tags in both files (18):

- `Bathymetry`, `GLOBAL_ID`, `MPAS_Parent`, `ROMS3DLocations`, `ROMS_Mask`, `ROMS_Salinity`, `ROMS_Temperature`, `ROMS_VelMeridional`, `ROMS_VelZonal`, `SSH`, `bathymetry`, `coriolis`, `salinityPistonVelocity`, `salinitySurfaceRestoringValue`, `temperaturePistonVelocity`, `temperatureSurfaceRestoringValue`, `windStressMeridional`, `windStressZonal`

Per-file tag differences:

- Only in BoB: `GRID_IMASK` (`int32`)
- Only in niskine: `mask` (`float64`)

Vector tags (`[100] float64` per cell):

- `ROMS3DLocations`
- `ROMS_Salinity`
- `ROMS_Temperature`
- `ROMS_VelMeridional`
- `ROMS_VelZonal`

## 4) Key 3D Field Characterization

### 4.1 `ROMS3DLocations` (100-level depth-like coordinate)

- BoB:
  - range `[-0.43824, 3511.51883]`, mean `423.08509`, p95 `2002.60160`
  - surface mean `0.02070`, deepest mean `1857.72251`
  - zero fraction `0.14239`
  - strictly increasing rows fraction `0.85761` overall; `1.0` on ocean cells (`ROMS_Mask==1`)
- niskine:
  - range `[0.99335, 3101.44604]`, mean `613.21684`, p95 `2273.07262`
  - surface mean `1.21527`, deepest mean `2701.64569`
  - zero fraction `0.0`
  - strictly increasing rows fraction `1.0`

### 4.2 `ROMS_Temperature`

- BoB:
  - range `[0.0, 27.85387]`, mean `14.02693`, p95 `26.80278`
  - surface mean `22.60502`, deepest mean `4.73561`
  - zero fraction `0.14239`
  - ocean-only mean (`ROMS_Mask==1`): `16.35583`
- niskine:
  - range `[2.67473, 10.75739]`, mean `7.53205`, p95 `10.58581`
  - surface mean `9.70030`, deepest mean `2.83534`
  - zero fraction `0.0`

### 4.3 `ROMS_Salinity`

- BoB:
  - range `[0.0, 35.04785]`, mean `29.16900`, p95 `35.02948`
  - surface mean `27.48740`, deepest mean `29.52762`
  - zero fraction `0.14239`
  - ocean-only mean (`ROMS_Mask==1`): `34.01194`
- niskine:
  - range `[34.90534, 35.36646]`, mean `35.11000`, p95 `35.33192`
  - surface mean `35.19900`, deepest mean `34.96130`
  - zero fraction `0.0`

### 4.4 `ROMS_VelMeridional`

- BoB: range `[-0.32668, 0.30474]`, mean `0.000459`, p95 `0.06673`, zero fraction `0.14239`
- niskine: range `[-0.33656, 0.37761]`, mean `0.013826`, p95 `0.23347`, zero fraction `0.0`

### 4.5 `ROMS_VelZonal`

- BoB: range `[-0.29484, 0.42543]`, mean `0.000142`, p95 `0.06267`, zero fraction `0.14239`
- niskine: range `[-0.28783, 0.57089]`, mean `0.024893`, p95 `0.24186`, zero fraction `0.0`

### 4.6 Combined Horizontal Speed (`sqrt(u^2 + v^2)`)

- BoB:
  - all-level mean `0.03786`, max `0.42585`
  - surface mean `0.09252`, surface p95 `0.27092`
  - deepest mean `0.00947`, deepest p95 `0.02987`
- niskine:
  - all-level mean `0.12307`, max `0.64307`
  - surface mean `0.19968`, surface p95 `0.47228`
  - deepest mean `0.04760`, deepest p95 `0.10779`

## 5) Scalar Field Highlights

BoB:

- `ROMS_Mask` mean `0.85761` (68,921 ocean cells, 11,443 land cells)
- `GRID_IMASK` mean `0.84957`
- `SSH` range `[0.0, 0.46767]`, mean `0.35738`
- `coriolis` range `[2.4701e-05, 5.9912e-05]`
- `bathymetry` range `[5.0, 3573.90943]`, mean `1895.25005`

niskine:

- `ROMS_Mask` is all ones
- `mask` is all ones (float scalar field)
- `SSH` range `[-0.88927, -0.49404]`, mean `-0.71877`
- `coriolis` range `[1.2194e-04, 1.2541e-04]`
- `bathymetry` range `[1373.17050, 3903.92196]`, mean `2754.11097`

Shared scalar behavior:

- `salinityPistonVelocity` and `temperaturePistonVelocity` are constant in niskine and mask-gated in BoB.
- Both files include both `Bathymetry` and `bathymetry` as distinct tags.

## 6) Parser Checklist

1. Read Quad4 mesh sizes from `/tstt/elements/Quad4/connectivity` and `/tstt/nodes/coordinates`.
2. Treat 3D ROMS fields as fixed-length 100-level vectors per cell.
3. Use mask-aware parsing:
   - BoB: `ROMS_Mask==0` rows are all-zero profiles for all five vector tags.
   - niskine: `ROMS_Mask==1` everywhere (no all-zero vector rows).
4. Handle file-specific mask tag naming:
   - BoB: `GRID_IMASK` (int)
   - niskine: `mask` (float)
5. Keep tag names case-sensitive (`Bathymetry` and `bathymetry` both exist).
6. Do not assume `/tstt/tags` definitions are all attached to Quad4; this group also contains inherited MPAS-oriented tag definitions.

## 7) Large Differences Between the Two Files

### 7.1 Resolution / Size

- BoB is ~`3.67x` larger by file size, node count, and cell count.
- Ratios:
  - file size: `3.6689x`
  - nodes: `3.6521x`
  - cells: `3.6750x`

### 7.2 Coverage and Masking

- BoB contains substantial masked land area:
  - 11,443 cells (`14.24%`) with `ROMS_Mask==0`
  - those rows are zeroed for all five vector fields.
- niskine is fully ocean-covered in this export (`ROMS_Mask==1` for all cells).

### 7.3 Tag-Level Differences

- Only BoB has `GRID_IMASK`.
- Only niskine has `mask`.

### 7.4 Field Regime Differences

- Temperature regime:
  - BoB is much warmer overall (ocean-only mean ~`16.36`) than niskine (mean ~`7.53`).
- Salinity regime:
  - niskine is tightly clustered around ~`35.11` with a narrow range.
  - BoB has wider variability and includes zeroed land rows.
- Velocity magnitude:
  - niskine has stronger flows (surface speed mean `0.19968` vs `0.09252` in BoB).
