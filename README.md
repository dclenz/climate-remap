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

## Generated Files

Remap runs write files such as `remap.mfa`, `remap_out.vtk`, `source_subset.vtk`, `reg-strength-*.txt`, and temporary `DIY.*` files in the current working directory. Run experiments from a dedicated output or scratch directory to keep the source checkout clean.
