if (CMAKE_VERSION VERSION_LESS "3.9")
  message(FATAL_ERROR "Diy requires CMake 3.9+")
endif()


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was diy-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(threads "ON")
set(log "OFF")
set(caliper "OFF")
set(mpi "ON")

include("${CMAKE_CURRENT_LIST_DIR}/diy-targets.cmake")

set(_diy_find_quietly)
if (${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  set(_diy_find_quietly QUIET)
endif()

if (threads)
  find_package(Threads ${_diy_find_quietly})
  if (NOT Threads_FOUND)
    list(APPEND "${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE" "Threads not found")
    set("${CMAKE_FIND_PACKAGE_NAME}_FOUND" 0)
  endif()
endif()

if (log)
  find_package(spdlog ${_diy_find_quietly})
  if (NOT spdlog_FOUND)
    list(APPEND "${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE" "SPDLOG not found")
    set("${CMAKE_FIND_PACKAGE_NAME}_FOUND" 0)
  endif()
endif()

if (caliper)
  find_package(caliper ${_diy_find_quietly})
  if (NOT caliper_FOUND)
    list(APPEND "${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE" "Caliper not found")
    set("${CMAKE_FIND_PACKAGE_NAME}_FOUND" 0)
  else()
    target_include_directories(DIY::diy INTERFACE $<INSTALL_INTERFACE:${caliper_INCLUDE_DIR}>)
  endif()
endif()

if (mpi)
  find_package(MPI ${_diy_find_quietly})
  if (NOT MPI_FOUND)
    list(APPEND "${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE" "MPI not found")
    set("${CMAKE_FIND_PACKAGE_NAME}_FOUND" 0)
  endif()
endif()

if (NOT DEFINED "${CMAKE_FIND_PACKAGE_NAME}_FOUND")
  set("${CMAKE_FIND_PACKAGE_NAME}_FOUND" 1)
endif ()
