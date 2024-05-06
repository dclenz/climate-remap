# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

class MfaRemap(CMakePackage):
    """Example of remapping two simulations with MFA"""

#     homepage = "https://github.com/dclenz/climate-remap"
#     url      = "https://github.com/dclenz/climate-remap"
#     git      = "https://github.com/dclenz/climate-remap.git"
    homepage = "https://github.com/tpeterka/climate-remap"
    url      = "https://github.com/tpeterka/climate-remap"
    git      = "https://github.com/tpeterka/climate-remap.git"

    version('master', branch='master')

    variant('build_type', default='Release', description='CMake build type', values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'))

    depends_on('mfa~examples~tests')
    depends_on('mpich')
    depends_on('hdf5+mpi+hl', type='link')
    depends_on('highfive')

    def cmake_args(self):
        args = ['-DCMAKE_BUILD_TYPE=%s' % self.spec.variants['build_type'].value,
                '-DCMAKE_C_COMPILER=%s' % self.spec['mpich'].mpicc,
                '-DCMAKE_CXX_COMPILER=%s' % self.spec['mpich'].mpicxx,
                '-DBUILD_SHARED_LIBS=false']
        return args

