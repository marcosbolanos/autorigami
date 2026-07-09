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
