# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dlenz/workspace/mfa/moab-example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dlenz/workspace/mfa/moab-example

# Utility rule file for ExperimentalMemCheck.

# Include any custom commands dependencies for this target.
include ext/diy/CMakeFiles/ExperimentalMemCheck.dir/compiler_depend.make

# Include the progress variables for this target.
include ext/diy/CMakeFiles/ExperimentalMemCheck.dir/progress.make

ext/diy/CMakeFiles/ExperimentalMemCheck:
	cd /home/dlenz/workspace/mfa/moab-example/ext/diy && /usr/bin/ctest -D ExperimentalMemCheck

ExperimentalMemCheck: ext/diy/CMakeFiles/ExperimentalMemCheck
ExperimentalMemCheck: ext/diy/CMakeFiles/ExperimentalMemCheck.dir/build.make
.PHONY : ExperimentalMemCheck

# Rule to build all files generated by this target.
ext/diy/CMakeFiles/ExperimentalMemCheck.dir/build: ExperimentalMemCheck
.PHONY : ext/diy/CMakeFiles/ExperimentalMemCheck.dir/build

ext/diy/CMakeFiles/ExperimentalMemCheck.dir/clean:
	cd /home/dlenz/workspace/mfa/moab-example/ext/diy && $(CMAKE_COMMAND) -P CMakeFiles/ExperimentalMemCheck.dir/cmake_clean.cmake
.PHONY : ext/diy/CMakeFiles/ExperimentalMemCheck.dir/clean

ext/diy/CMakeFiles/ExperimentalMemCheck.dir/depend:
	cd /home/dlenz/workspace/mfa/moab-example && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dlenz/workspace/mfa/moab-example /home/dlenz/workspace/mfa/moab-example/ext/diy /home/dlenz/workspace/mfa/moab-example /home/dlenz/workspace/mfa/moab-example/ext/diy /home/dlenz/workspace/mfa/moab-example/ext/diy/CMakeFiles/ExperimentalMemCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ext/diy/CMakeFiles/ExperimentalMemCheck.dir/depend

