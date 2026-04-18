include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(BUILD_TESTING OFF CACHE BOOL "Disable third-party dependency tests" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)

set(AUTORIGAMI_EIGEN_VERSION "3.4.0")
set(AUTORIGAMI_EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz")
set(AUTORIGAMI_EIGEN_SHA256 "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")

set(AUTORIGAMI_GEOMETRY_CENTRAL_TAG "v1.0.0")
set(AUTORIGAMI_GEOMETRY_CENTRAL_COMMIT "1f8a50c353e90322294c1c5aa0d411b7894f24ed")
set(AUTORIGAMI_GEOMETRY_CENTRAL_REPOSITORY "https://github.com/nmwsharp/geometry-central.git")

set(AUTORIGAMI_BOOST_VERSION "1.90.0")
set(AUTORIGAMI_BOOST_URL
    "https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.gz")
set(AUTORIGAMI_BOOST_SHA256 "5e93d582aff26868d581a52ae78c7d8edf3f3064742c6e77901a1f18a437eea9")

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

if(NOT TARGET Eigen3::Eigen)
    add_library(autorigami_eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS autorigami_eigen)
    target_include_directories(
        autorigami_eigen
        SYSTEM
        INTERFACE
        "${eigen_SOURCE_DIR}"
    )
endif()

set(SUITESPARSE OFF CACHE BOOL "Disable optional SuiteSparse in geometry-central" FORCE)
set(GC_ALWAYS_DOWNLOAD_EIGEN OFF CACHE BOOL "Use autorigami-pinned Eigen target" FORCE)

FetchContent_Declare(
    geometry_central
    GIT_REPOSITORY "${AUTORIGAMI_GEOMETRY_CENTRAL_REPOSITORY}"
    GIT_TAG "${AUTORIGAMI_GEOMETRY_CENTRAL_COMMIT}"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
    GIT_SUBMODULES "deps/happly"
)

FetchContent_MakeAvailable(geometry_central)

FetchContent_Declare(
    boost
    URL "${AUTORIGAMI_BOOST_URL}"
    URL_HASH "SHA256=${AUTORIGAMI_BOOST_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(boost)

if(NOT TARGET autorigami_boost)
  add_library(autorigami_boost INTERFACE)
  target_include_directories(
      autorigami_boost
      SYSTEM
      INTERFACE
      "${boost_SOURCE_DIR}"
  )
endif()
