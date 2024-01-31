# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

class MfaRemap(CMakePackage):
    """Example of remapping two simulations with MFA"""

    homepage = "https://github.com/dclenz/climate-remap"
    url      = "https://github.com/dclenz/climate-remap"
    git      = "https://github.com/dclenz/climate-remap.git"

    version('master', branch='master')

    depends_on('mpich')
    depends_on('hdf5+mpi+hl', type='link')

    def cmake_args(self):
        args = ['-DCMAKE_C_COMPILER=%s' % self.spec['mpich'].mpicc,
                '-DCMAKE_CXX_COMPILER=%s' % self.spec['mpich'].mpicxx,
                '-DBUILD_SHARED_LIBS=false']
        return args
