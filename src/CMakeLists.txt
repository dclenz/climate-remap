if                          (NOT APPLE)
   set                      (linker_flags "-pie -Wl,--export-dynamic")
   set                      (linker_flags "${linker_flags} -Wl,-u,henson_set_contexts,-u,henson_set_namemap")
else                        ()
   set                      (linker_flags "-Wl,-u,_henson_set_contexts,-u,_henson_set_namemap")
endif                       (NOT APPLE)

include_directories         (${CMAKE_SOURCE_DIR}/src)          # needed for block.hpp and opts.h

add_executable              (mfa_remap              remap.cpp)
target_link_libraries       (mfa_remap              ${libraries})
set_property                (TARGET mfa_remap       PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable              (write_vtk              write_vtk.cpp writer.cpp)
target_link_libraries       (write_vtk              ${libraries})
set_property                (TARGET write_vtk       PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library                 (mfa_remap_shared SHARED remap.cpp)
target_link_libraries       (mfa_remap_shared ${libraries})
set_target_properties       (mfa_remap_shared PROPERTIES PREFIX "")
set_target_properties       (mfa_remap_shared PROPERTIES SUFFIX ".so")
set_target_properties       (mfa_remap_shared PROPERTIES LINK_FLAGS ${linker_flags})

install                     (TARGETS mfa_remap write_vtk mfa_remap_shared
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_WRITE GROUP_EXECUTE
                            WORLD_READ WORLD_WRITE WORLD_EXECUTE)

