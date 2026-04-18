include(cmake/Dependencies.cmake)

find_package(pybind11 CONFIG REQUIRED)

option(AUTORIGAMI_BUILD_TESTS "Build native C++ tests" ON)

pybind11_add_module(
    _native
    MODULE
    src/bindings/module.cpp
    src/bindings/validation_bindings.cpp
    src/splines/splines.cpp
    src/validation/validation.cpp
    src/curvature/curvature_math.cpp
)

target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(_native PRIVATE geometry-central)

install(TARGETS _native DESTINATION autorigami COMPONENT autorigami-python)

if(AUTORIGAMI_BUILD_TESTS)
    add_executable(
        autorigami_validation_tests
        tests/validation/validation_test.cpp
        src/splines/splines.cpp
        src/curvature/curvature_math.cpp
        src/validation/validation.cpp
    )

    target_compile_features(autorigami_validation_tests PRIVATE cxx_std_20)
    target_include_directories(autorigami_validation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(autorigami_validation_tests PRIVATE geometry-central)

    add_test(NAME autorigami_validation_tests COMMAND autorigami_validation_tests)

    add_executable(
        autorigami_splines_tests
        tests/splines/splines_test.cpp
        src/splines/splines.cpp
    )

    target_compile_features(autorigami_splines_tests PRIVATE cxx_std_20)
    target_include_directories(autorigami_splines_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(autorigami_splines_tests PRIVATE geometry-central)

    add_test(NAME autorigami_splines_tests COMMAND autorigami_splines_tests)

    add_executable(
        autorigami_curvature_math_tests
        tests/curvature/curvature_math_test.cpp
        src/splines/splines.cpp
        src/curvature/curvature_math.cpp
    )

    target_compile_features(autorigami_curvature_math_tests PRIVATE cxx_std_20)
    target_include_directories(autorigami_curvature_math_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(autorigami_curvature_math_tests PRIVATE geometry-central)

    add_test(NAME autorigami_curvature_math_tests COMMAND autorigami_curvature_math_tests)
endif()
