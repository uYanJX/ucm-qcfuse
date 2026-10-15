file(GLOB_RECURSE UCMINFRA_STATUS_SOURCE_FILES "status/*.*")
add_library(infra_status OBJECT ${UCMINFRA_STATUS_SOURCE_FILES})
target_include_directories(infra_status PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infra_status PUBLIC fmt)

add_subdirectory(logger)

# These directories currently contain headers only.  INTERFACE targets express
# that fact and keep modern CMake from trying to infer a linker language for an
# empty object library while preserving the include propagation used by stores.
add_library(infra_template INTERFACE)
target_include_directories(infra_template INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(infra_thread INTERFACE)
target_include_directories(infra_thread INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(infra_time INTERFACE)
target_include_directories(infra_time INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
