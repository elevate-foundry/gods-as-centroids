cmake_minimum_required(VERSION 3.16)
project(gods-as-centroids)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(nlohmann_json QUIET)

# Add executable
add_executable(gabm
    src/main.cpp
    src/kernel.cpp
    src/agent.cpp
    src/utils.cpp
)

# Include directories
target_include_directories(gabm PRIVATE src)

# Link libraries
if(nlohmann_json_FOUND)
    target_link_libraries(gabm nlohmann_json::nlohmann_json)
else()
    # Fallback to header-only json library
    target_include_directories(gabm PRIVATE third_party)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(gabm PRIVATE -O3 -march=native -flto)
    target_link_options(gabm PRIVATE -flto)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(gabm PRIVATE /O2 /GL)
    target_link_options(gabm PRIVATE /LTCG)
endif()
