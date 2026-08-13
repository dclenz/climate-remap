# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack.package import *


class Mfa(CMakePackage):
    """Multivariate functional approximation library."""

    homepage = "https://github.com/tpeterka/mfa"
    git = "https://github.com/tpeterka/mfa.git"

    version("master", branch="master")

    variant(
        "build_type",
        default="Release",
        description="CMake build type",
        values=("Debug", "Release", "RelWithDebInfo", "MinSizeRel"),
    )
    variant(
        "thread",
        values=("serial", "tbb", "sycl", "kokkos"),
        default="serial",
        description="Threading model",
    )
    variant("examples", default=True, description="Build MFA examples")
    variant("tests", default=True, description="Build MFA unit tests")

    depends_on("mpi", type=("build", "link"))
    depends_on("tbb", when="thread=tbb")
    depends_on("kokkos", when="thread=kokkos")

    def cmake_args(self):
        return [
            self.define("CMAKE_C_COMPILER", self.spec["mpi"].mpicc),
            self.define("CMAKE_CXX_COMPILER", self.spec["mpi"].mpicxx),
            self.define_from_variant("mfa_build_examples", "examples"),
            self.define_from_variant("mfa_build_tests", "tests"),
            self.define_from_variant("mfa_thread", "thread"),
        ]
