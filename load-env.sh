#!/bin/bash

# activate the environment
export SPACKENV=mfa-remap-env
spack env deactivate > /dev/null 2>&1
spack env activate $SPACKENV
echo "activated spack environment $SPACKENV"

echo "setting flags for building mfa-remap"
export MFA_PATH=`spack location -i mfa`
export HIGHFIVE_PATH=`spack location -i highfive`
export MOAB_PATH=`spack location -i moab`
export MFA_REMAP_PATH=`spack location -i mfa-remap`
export HENSON_PATH=`spack location -i henson`

echo "setting flags for running moab-example"
export LD_LIBRARY_PATH=$MOAB_PATH/lib:$LD_LIBRARY_PATH


