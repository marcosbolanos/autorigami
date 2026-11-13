include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(BUILD_TESTING OFF CACHE BOOL "Disable third-party dependency tests" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build dependencies as PIC for shared modules" FORCE)

set(AUTORIGAMI_EIGEN_VERSION "3.4.0")
set(AUTORIGAMI_EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz")
set(AUTORIGAMI_EIGEN_SHA256 "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")

FetchContent_Declare(
    eigen
    URL "${AUTORIGAMI_EIGEN_URL}"
    URL_HASH "SHA256=${AUTORIGAMI_EIGEN_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_MakeAvailable(eigen)
endif()

# PhysX SDK (BSD-3-Clause source), built as CPU-only static libraries.
set(AUTORIGAMI_PHYSX_COMMIT "b97365c6ddb32fda8469bc788ec18024862b0368")

FetchContent_Declare(
    physx
    URL "https://github.com/NVIDIA-Omniverse/PhysX/archive/${AUTORIGAMI_PHYSX_COMMIT}.tar.gz"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # The repository root has no CMakeLists.txt; the SDK is added below.
    SOURCE_SUBDIR "not-a-cmake-directory"
)
FetchContent_MakeAvailable(physx)

set(PHYSX_ROOT_DIR "${physx_SOURCE_DIR}/physx")
# PhysX has build files for Windows and Linux only. The Linux files also work
# with Apple Clang, thus macOS uses them.
if(WIN32)
    set(TARGET_BUILD_PLATFORM "windows")
else()
    set(TARGET_BUILD_PLATFORM "linux")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(PX_OUTPUT_ARCH "arm")
else()
    set(PX_OUTPUT_ARCH "x86")
endif()
set(PX_OUTPUT_LIB_DIR "${CMAKE_BINARY_DIR}/physx-output")
set(PX_OUTPUT_BIN_DIR "${CMAKE_BINARY_DIR}/physx-output")
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "Build PhysX as static libraries" FORCE)
set(PX_BUILDSNIPPETS OFF CACHE BOOL "Skip PhysX snippets" FORCE)
set(PX_BUILDPVDRUNTIME OFF CACHE BOOL "Skip the OmniPVD runtime" FORCE)
set(PX_GENERATE_GPU_PROJECTS OFF CACHE BOOL "Skip PhysX GPU projects" FORCE)
add_subdirectory(
    "${PHYSX_ROOT_DIR}/compiler/public"
    "${CMAKE_BINARY_DIR}/physx-build"
    EXCLUDE_FROM_ALL
)

# PhysX compiles with warnings as errors, which fails on compilers newer than
# the ones that NVIDIA tests. Target options come after the PhysX flags, so
# this wins.
if(MSVC)
    set(AUTORIGAMI_PHYSX_NO_WARNING_ERRORS /WX-)
else()
    set(AUTORIGAMI_PHYSX_NO_WARNING_ERRORS -Wno-error)
endif()
function(autorigami_relax_physx_warnings directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type "${target}" TYPE)
        if(NOT type STREQUAL "INTERFACE_LIBRARY" AND NOT type STREQUAL "UTILITY")
            target_compile_options(
                "${target}" PRIVATE ${AUTORIGAMI_PHYSX_NO_WARNING_ERRORS}
            )
        endif()
    endforeach()
    get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
        autorigami_relax_physx_warnings("${subdirectory}")
    endforeach()
endfunction()
autorigami_relax_physx_warnings("${PHYSX_ROOT_DIR}/compiler/public")
