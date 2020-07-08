cmake_minimum_required(VERSION 3.0)
project(ilqr)

# --- [ External libs ]---

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS OFF CACHE BOOL "" FORCE)

# EXCLUDE_FROM_ALL is used here to prevent execution of the
# install targets of these subdirectories
add_subdirectory(extern/eigen EXCLUDE_FROM_ALL)

# Build the main executable

add_executable(${PROJECT_NAME} main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_compile_options(${PROJECT_NAME} PUBLIC
    -Wall
    -Wextra
    -Wpedantic
    -Wunreachable-code
    -std=c++11
    -fPIC
    -fopenmp

    # For the really paranoid
    #-Wconversion 
)

target_link_libraries(${PROJECT_NAME}
                      pthread
                      rt
                      gomp)

# Compile type dependent (release or debug) flags.

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PUBLIC -g)
    target_compile_options(${PROJECT_NAME} PUBLIC -O0)
else()
    target_compile_options(${PROJECT_NAME} PUBLIC -O3)
    target_compile_options(${PROJECT_NAME} PUBLIC -mfpmath=sse)
endif()

# Exports compile commands to .json file for vim YouCompleteMe support.

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
