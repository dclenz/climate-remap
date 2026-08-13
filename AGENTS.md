# Agent Guidance

## Build Environment

This project uses a managed Spack environment and Spack develop packages.

Initial installation is execution-only:

```bash
./create-env.sh
```

Do not source `create-env.sh`. It creates or updates `mfa-remap-env`, registers this checkout as the environment-local package repository, uses this checkout as the `mfa-remap` develop source, configures MFA in develop mode, concretizes, and installs the environment.

For every shell used to build or run the project, source the activation script:

```bash
source load-env.sh
```

Do not execute `load-env.sh`; activation and its exported variables must persist in the current shell. The important variables are:

- `MFA_REMAP_ROOT`: project checkout
- `MFA_REMAP_PATH`: installed `mfa-remap` prefix
- `SPACKENV`: active Spack environment name

After source changes, rebuild and install with:

```bash
source load-env.sh
spack install
```

Use the installed executables under `$MFA_REMAP_PATH/bin`. In particular:

```bash
$MFA_REMAP_PATH/bin/mfa_remap
```

Do not run a stale executable from `build/` or another local build directory when validating current source changes.

The scripts find Spack through `SPACK_ROOT`, `spack` on `PATH`, or `$HOME/spack`. They derive the checkout path from the script location; do not add machine-specific absolute paths.

`SPACK_ENV_NAME` may override the default environment name. Use the same value for `create-env.sh` and `load-env.sh`.

## Running Experiments

The application writes model, VTK, regularization, and DIY files into its current working directory. Run tests from a dedicated scratch/output directory rather than the repository root. Large runs can exceed normal command timeouts, so set an explicit longer timeout and preserve the run log.

Canonical source and target data used during development are:

```text
data/mpas_grid_tstep0.h5m
data/roms_3d_projected_BoB.h5m
```

These files are large and may not exist in a fresh clone. Do not modify or regenerate them unless the task explicitly requires it.
