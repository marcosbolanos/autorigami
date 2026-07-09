include(cmake/Dependencies.cmake)

find_package(pybind11 CONFIG REQUIRED)

set(
    AUTORIGAMI_CLANG_TIDY_CMD
    clang-tidy
    -p=${CMAKE_BINARY_DIR}
    --checks=clang-analyzer-*
    --header-filter=${CMAKE_SOURCE_DIR}/cpp/.*
    --warnings-as-errors=*
)

pybind11_add_module(
    _native
    MODULE
    src/bindings/module.cpp
    src/bindings/geometry_bindings.cpp
    src/geometry/geometry.cpp
)

target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_native PRIVATE Eigen3::Eigen)
if(AUTORIGAMI_DEV_LINT)
    set_property(TARGET _native PROPERTY CXX_CLANG_TIDY "${AUTORIGAMI_CLANG_TIDY_CMD}")
endif()

install(TARGETS _native DESTINATION autorigami COMPONENT autorigami-python)
