# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack.package import *


class MfaRemap(CMakePackage):
    """Remap climate data with multivariate functional approximation."""

    homepage = "https://github.com/dclenz/climate-remap"
    git = "https://github.com/dclenz/climate-remap.git"

    version("master", branch="master")

    variant(
        "build_type",
        default="RelWithDebInfo",
        description="CMake build type",
        values=("Debug", "Release", "RelWithDebInfo", "MinSizeRel"),
    )

    depends_on("mfa@master~examples~tests thread=tbb", type=("build", "link"))
    depends_on("mpi", type=("build", "link"))
    depends_on("hdf5@1.14+mpi+hl", type=("build", "link"))
    depends_on("highfive+mpi", type=("build", "link"))
    depends_on("boost", type="build")
    depends_on("henson@master+python+mpi-wrappers", type=("build", "link"))
    depends_on(
        "moab@5.5.0~cgm~coupler~dagmc~debug~fbigeom~fortran+hdf5~irel"
        "~metis+mpi~netcdf~parmetis~pnetcdf~shared~zoltan",
        type=("build", "link"),
    )

    def cmake_args(self):
        return [
            self.define("CMAKE_C_COMPILER", self.spec["mpi"].mpicc),
            self.define("CMAKE_CXX_COMPILER", self.spec["mpi"].mpicxx),
            self.define("MFA_PATH", self.spec["mfa"].prefix),
            self.define("HIGHFIVE_PATH", self.spec["highfive"].prefix),
            self.define("MFA_THREAD", self.spec["mfa"].variants["thread"].value),
            self.define("BUILD_SHARED_LIBS", False),
        ]

