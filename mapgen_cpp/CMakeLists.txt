cmake_minimum_required(VERSION 3.16)
project(mapgen CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored Recast/Detour sources.
set(RECAST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/recastnavigation")

file(GLOB RECAST_SOURCES
    "${RECAST_ROOT}/Recast/Source/*.cpp"
    "${RECAST_ROOT}/Detour/Source/*.cpp"
)

add_library(recast_detour STATIC ${RECAST_SOURCES})
target_include_directories(recast_detour PUBLIC
    "${RECAST_ROOT}/Recast/Include"
    "${RECAST_ROOT}/Detour/Include"
)

# Mapgen library (all sources except main.cpp).
file(GLOB MAPGEN_LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER MAPGEN_LIB_SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(mapgen_lib STATIC ${MAPGEN_LIB_SOURCES})
target_include_directories(mapgen_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(mapgen_lib PUBLIC recast_detour)

# Mapgen executable.
add_executable(mapgen "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(mapgen PRIVATE mapgen_lib)
