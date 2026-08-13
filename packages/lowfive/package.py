# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack.package import *


class Lowfive(CMakePackage):
    """HDF5 VOL Plugin for In Situ Data Transport."""

    # public repo
    homepage = "https://github.com/diatomic/LowFive.git"
    url      = "https://github.com/diatomic/LowFive.git"
    git      = "https://github.com/diatomic/LowFive.git"
    version('master', branch='master')
#     version('datatypes', branch='datatypes')

    variant("examples", default=False, description="Install the examples")
    variant("auto_load", default=True, description="Set LowFive environment variables")
    variant("python", default=True, description="Install Python bindings")

    depends_on("mpi", type=("build", "link"))
    depends_on("hdf5+mpi+hl", type=("build", "link"))

    extends("python", when="+python")       # brings pylowfive into PYTHONPATH
    depends_on("py-mpi4py", when="+python", type=("build", "run"))

    def cmake_args(self):
        args = [self.define("CMAKE_C_COMPILER", self.spec["mpi"].mpicc),
                self.define("CMAKE_CXX_COMPILER", self.spec["mpi"].mpicxx),
                self.define("BUILD_SHARED_LIBS", False),
                self.define_from_variant('lowfive_install_examples', 'examples'),
                self.define_from_variant('lowfive_python', 'python'),
        ]

        if self.spec.satisfies("+python"):
            args += [self.define("PYTHON_EXECUTABLE", self.spec["python"].command.path)]

        return args


    def setup_run_environment(self, env):
        if "+auto_load" in self.spec:
            env.set('HDF5_VOL_CONNECTOR', "lowfive under_vol=0;under_info={};")
            env.set('HDF5_PLUGIN_PATH', self.prefix.lib)
