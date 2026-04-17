include(cmake/Dependencies.cmake)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _native
    MODULE
    bindings/module.cpp
    bindings/validation_bindings.cpp
    validation/validation.cpp
)

target_compile_features(_native PRIVATE cxx_std_17)
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_native PRIVATE geometry-central)

install(TARGETS _native DESTINATION autorigami COMPONENT autorigami-python)
