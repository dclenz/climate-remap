project                     (mfa)
cmake_minimum_required      (VERSION 3.1)

option                      (wrapped_mpi    "MPI compiler wrapper requires no further MPI libs, true/false (default false)" OFF     )
option                      (mfa_thread     "tbb, kokkos, sycl, or serial (default serial)"                                 serial  )
option                      (s3d_infile     "Path/filename of s3d dataset (for unit testing)"                                       )
option                      (python         "Build Python bindings, true/false (default false)"                             OFF     )
option                      (eigen_thread   "Use OpenMP for Eigen threading, true/false (default false)"                    OFF     )
option                      (OPENMP_LIBRARY "OpenMP library if not found automatically (optional)"                                  )

# C++11
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Default to Release
if                          (NOT CMAKE_BUILD_TYPE)
    set                     (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    set_property            (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif                       (NOT CMAKE_BUILD_TYPE)

# MPI
if                          (NOT wrapped_mpi)
    find_package            (MPI REQUIRED)
    set                     (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       ()

# MFA threading model
if                          (mfa_thread MATCHES "tbb")
    message                 ("Using TBB threading")
    add_definitions         (-DMFA_TBB)
    find_package            (TBB QUIET)
    find_path               (TBB_INCLUDE_DIR        tbb/tbb.h)
    find_library            (TBB_LIBRARY NAMES      tbb)
    if                      (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        include_directories ("${TBB_INCLUDE_DIR}")
        set                 (libraries ${libraries} ${TBB_LIBRARY})
        message             ("Found TBB in ${TBB_INCLUDE_DIR} and in ${TBB_LIBRARY}")
    else                    (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        message             (FATAL_ERROR "Could not find TBB")
    endif                   ()
else                        ()
    message                 ("Using no threading (serial)")
    add_definitions         (-DMFA_SERIAL)
endif                       ()

# Eigen threading (requires OpenMP to be available)
if                          (eigen_thread)
    find_package            (OpenMP)
    message                 ("Using OpenMP for Eigen")
    add_definitions         (-DEIGEN_OPENMP)
    if                      (APPLE)
        set                 (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
        if                  (${OPENMP_CXX_FOUND})
            set             (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
        elseif              (DEFINED OPENMP_INCLUDE_DIRS AND DEFINED OPENMP_LIBRARY)
            include_directories (${OPENMP_INCLUDE_DIRS})
            set             (libraries ${libraries} ${OPENMP_LIBRARY})
            message         ("Using user-provided OpenMP headers and library: ${OPENMP_INCLUDE_DIRS} ${OPENMP_LIBRARY}")
        else                ()
            message         ("Error: cannot find OpenMP and OPENMP_INCLUDE_DIRS or OPENMP_LIBRARY is not set")
            message         (FATAL_ERROR "Either provide OPENMP_INCLUDE_DIRS and OPENMP_LIBRARY explicitly or disable eigen_thread")
        endif               ()
    else                    ()
        set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif                   ()
endif                       ()

# Threads
find_package                (Threads)

# HDF5
if                          (HDF5_LIBRARY AND HDF5_INCLUDE_DIR)
    message                 ("HDF5_LIBRARY = " ${HDF5_LIBRARY})
    message                 ("HDF5_INCLUDE_DIR = " ${HDF5_INCLUDE_DIR})
else                        ()
    message                 ("Looking for HDF5")
    find_package            (HDF5)
    set                     (HDF5_LIBRARY ${HDF5_C_LIBRARY_hdf5})
endif                       ()
set                         (libraries ${libraries} ${HDF5_LIBRARY})
include_directories         (SYSTEM ${HDF5_INCLUDE_DIR})

# zlib
find_package 		        (ZLIB)
set 			            (libraries ${libraries} ${ZLIB_LIBRARIES})

# HighFive
if                          (HighFive_INCLUDE_DIR)
    include_directories     (SYSTEM ${HighFive_INCLUDE_DIR})
    message                 ("HighFive_INCLUDE_DIR = " ${HighFive_INCLUDE_DIR})
endif                       ()

# MFA (top-level include; i.e. <path-to-mfa>/include, not <path-to-mfa>/include/mfa)
if                          (MFA_INCLUDE_DIR)
    message                 ("MFA_INCLUDE_DIR = " ${MFA_INCLUDE_DIR})
    add_definitions         (-DMFA_NO_WEIGHTS)
    message                 ("Weights are disabled for MFA")
else                        (MFA_INCLUDE_DIR)
    message                 (FATAL_ERROR "Could not find MFA")
endif                       ()

# Include dirs: mfa, cppoptlib, diy, eigen, mpi
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (SYSTEM ${MFA_INCLUDE_DIR}
                            ${MFA_INCLUDE_DIR}/cppoptlib/include
                            ${MFA_INCLUDE_DIR}/diy/include
                            ${MFA_INCLUDE_DIR}/eigen3
                            ${MPI_INCLUDE_PATH}
                            )

# Libraries
set                         (libraries
                             ${libraries}
                             ${CMAKE_DL_LIBS} 
                             ${CMAKE_THREAD_LIBS_INIT} )


# Subdirectories
add_subdirectory            (src)
if                          (python)
    add_subdirectory        (include/pybind11)
    add_subdirectory        (python)
endif                       ()

# Install the headers
file                        (GLOB DEPLOY_FILES_AND_DIRS "${PROJECT_SOURCE_DIR}/include/*")
foreach                     (ITEM ${DEPLOY_FILES_AND_DIRS})
   if                       (IS_DIRECTORY "${ITEM}")
      list                  (APPEND DIRS_TO_DEPLOY "${ITEM}")
   else                     ()
      list                  (APPEND FILES_TO_DEPLOY "${ITEM}")
   endif                    ()
endforeach                  ()
install                     (FILES ${FILES_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install                     (DIRECTORY ${DIRS_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

