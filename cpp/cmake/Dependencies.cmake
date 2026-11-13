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

# PhysX has no macOS support. Three source files need a small change for
# Apple Clang. Each change is a text replacement, applied one time.
function(autorigami_patch_physx file old new)
    set(path "${PHYSX_ROOT_DIR}/source/${file}")
    file(READ "${path}" content)
    string(FIND "${content}" "${new}" patched)
    if(patched EQUAL -1)
        string(FIND "${content}" "${old}" found)
        if(found EQUAL -1)
            message(FATAL_ERROR "PhysX patch does not apply to ${file}")
        endif()
        string(REPLACE "${old}" "${new}" content "${content}")
        file(WRITE "${path}" "${content}")
    endif()
endfunction()
if(APPLE)
    # size_t is not uint32_t or uint64_t on macOS, thus the call is ambiguous.
    autorigami_patch_physx(
        "foundation/FdTempAllocator.cpp"
        "PxHighestSetBit(PxU32(size) + sizeof(Chunk) - 1)"
        "PxHighestSetBit(PxU32(PxU32(size) + sizeof(Chunk) - 1))"
    )
    # The serialization tag list has no entry for macOS on arm64.
    autorigami_patch_physx(
        "physxextensions/src/serialization/SnSerialUtils.cpp"
        "#elif PX_LINUX && PX_A64"
        "#elif (PX_LINUX || PX_OSX) && PX_A64"
    )
    # The macOS floating-point code uses SSE, which arm64 does not have.
    autorigami_patch_physx(
        "foundation/unix/FdUnixFPU.cpp"
        "PX_OSX"
        "(PX_OSX && !PX_A64)"
    )
endif()
# PhysX has build files for Windows and Linux only. macOS uses the Linux
# files.
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
