#!/bin/bash

export SPACKENV=mfa-remap-env
export YAML=$PWD/env.yaml

# create spack environment
echo "creating spack environment $SPACKENV"
spack env deactivate > /dev/null 2>&1
spack env remove -y $SPACKENV > /dev/null 2>&1
spack env create $SPACKENV $YAML

# activate environment
echo "activating spack environment"
spack env activate $SPACKENV

# add mfa in develop mode
spack develop mfa@master~tests~examples build_type=RelWithDebInfo
spack add mfa

# add mfa-remap in develop mode (point at this local checkout)
spack develop --path $PWD mfa-remap@master build_type=RelWithDebInfo
spack add mfa-remap

spack add henson+python+mpi-wrappers

# install everything in environment
echo "installing dependencies in environment"
spack install mfa
spack install highfive
spack install moab
spack install henson

# set build flags
echo "setting flags for building mfa-remap"
export MFA_PATH=`spack location -i mfa`
export HIGHFIVE_PATH=`spack location -i highfive`
export HENSON_PATH=`spack location -i henson`

echo "installing mfa-remap"
spack install mfa-remap
export MFA_REMAP_PATH=`spack location -i mfa-remap`

# reset the environment (workaround for spack behavior)
spack env deactivate
spack env activate $SPACKENV


