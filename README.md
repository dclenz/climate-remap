# Climate Remap

Climate Remap remaps MPAS climate fields onto a ROMS mesh using multivariate functional approximation (MFA). The project is built and installed with Spack.

## Prerequisites

- A working Spack installation
- A C, C++, and Fortran compiler discoverable by Spack

The environment scripts locate Spack in this order:

1. `$SPACK_ROOT/share/spack/setup-env.sh`
2. A `spack` executable already on `PATH`
3. `$HOME/spack/share/spack/setup-env.sh`

If Spack is installed elsewhere and is not on `PATH`, set `SPACK_ROOT` before using the scripts.

## Initial Installation

Run the installer from the project checkout:

```bash
./create-env.sh
```

`create-env.sh` must be executed, not sourced. It:

- Creates or updates the managed environment `mfa-remap-env`.
- Registers this checkout as an environment-local Spack package repository.
- Configures this checkout as the `mfa-remap` develop source.
- Configures MFA in develop mode.
- Selects MPICH 4 as the environment's MPI provider.
- Concretizes and installs `mfa-remap` and its dependencies.

The installer derives the project path from its own location, so it can also be invoked with an absolute path from another directory.

## Activate The Environment

For each new shell, source the activation script:

```bash
source load-env.sh
```

`load-env.sh` must be sourced, not executed, because Spack activation and exported variables must modify the current shell. It exports:

- `SPACKENV`: the active managed environment name.
- `MFA_REMAP_ROOT`: the project checkout path.
- `MFA_REMAP_PATH`: the installed `mfa-remap` prefix.

Executables are installed under:

```bash
$MFA_REMAP_PATH/bin
```

For example:

```bash
$MFA_REMAP_PATH/bin/mfa_remap --help
```

## Rebuild After Source Changes

After activating the environment, rebuild and install the develop checkout with:

```bash
source load-env.sh
spack install
```

Run the installed executable from `$MFA_REMAP_PATH/bin`. Do not assume an older executable in a local build directory contains the latest source changes.

## Alternate Environment Name

The scripts use `mfa-remap-env` by default. To use another managed environment, set the same `SPACK_ENV_NAME` for installation and activation:

```bash
SPACK_ENV_NAME=my-mfa-remap ./create-env.sh
export SPACK_ENV_NAME=my-mfa-remap
source load-env.sh
```

## Round-Trip Error Analysis

`remap_loop` performs an MPAS-to-ROMS remap followed by a higher-resolution ROMS-to-MPAS remap. It decodes at the original active MPAS cell centroids and reports pointwise round-trip errors.

Run it from a dedicated output directory because generated files are written to the current working directory:

```bash
source load-env.sh
mkdir -p runs/bob
cd runs/bob

"$MFA_REMAP_PATH/bin/remap_loop" \
  --mpasfile "$MFA_REMAP_ROOT/data/mpas_grid_tstep0.h5m" \
  --romsfile "$MFA_REMAP_ROOT/data/roms_3d_projected_BoB.h5m" \
  2>&1 | tee run.log
```

The default forward and reverse control lattices are `11 x 11 x 11` and `17 x 17 x 17`. Repeat the corresponding option three times to set each dimension independently:

```bash
--vars_nctrl 35 --vars_nctrl 35 --vars_nctrl 50 \
--reverse_vars_nctrl 80 --reverse_vars_nctrl 80 --reverse_vars_nctrl 70
```

Forward and reverse regularization are independent:

```bash
--regularization 1 --reverse_regularization 1
```

`MFA_NUM_THREADS` limits TBB worker concurrency when needed:

```bash
MFA_NUM_THREADS=4 "$MFA_REMAP_PATH/bin/remap_loop" ...
```

Use `remap_loop --help` for the complete option list.

### Round-Trip Outputs

`remap_loop` writes:

- `forward_remap.h5m`: the ROMS mesh with complete forward-remapped profiles.
- `roundtrip_mpas.h5m`: the selected MPAS mesh with round-trip fields, errors, coverage, and eligibility.
- `roundtrip_metrics.csv`: overall and per-level bias, RMSE, maxima, and value ranges.
- `forward_remap.mfa` and `reverse_remap.mfa`: serialized MFA models and point data.

The primary MPAS filtering tags in `roundtrip_mpas.h5m` are:

- `roundtrip_error_cell_eligible`: scalar cell flag; `1` means at least one active level from the cell contributes to error metrics, otherwise `0`.
- `roundtrip_error_eligible[k]`: per-level flag for MPAS level `k`.
- `roundtrip_coverage_fraction`: fraction of the projected MPAS polygon covered by wet ROMS cells.
- `roundtrip_coverage_class`: `0` uncovered, `1` partial, `2` fully covered, and `3` halo.

Load `/Users/dlenz/workspace/moab_plugin/moab_h5m_reader.py` in ParaView 5.10, open `roundtrip_mpas.h5m`, and threshold `roundtrip_error_cell_eligible == 1` to display cells used by the error analysis. For level-specific plots, also threshold `roundtrip_error_eligible[k] == 1`.

## Generated Files

The legacy `mfa_remap` executable writes files such as `remap.mfa`, `remap_out.vtk`, `source_subset.vtk`, `reg-strength-*.txt`, and temporary `DIY.*` files in the current working directory. Run experiments from a dedicated output or scratch directory to keep the source checkout clean.
