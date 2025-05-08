cmake_minimum_required(VERSION 3.31)
project(EM_regularization)

find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_STANDARD 17)

add_executable(EM_regularization main.cpp
        data.cpp
        algorithm/gmm/gmm.cpp
        algorithm/gmm/gmmResult.h
        algorithm/gmm/helpers/sampling.cpp
        algorithm/gmm/helpers/mathematical.cpp
        algorithm/regularization/CovarianceMatrixRegularizer.cpp
        algorithm/regularization/CovarianceMatrixRegularizer.h
        algorithm/regularization/EmpiricalRegularizer.cpp
        algorithm/gmm/gmm.h
        algorithm/GeneticalAlgorithm.h
        algorithm/GeneticalAlgorithm.cpp
        external/hungarian/hungarian.cpp
        external/hungarian/hungarian.h)

enable_testing()
find_package(GTest REQUIRED)

add_executable(gmm_tests
        Testing/test_gmm.cpp
        Testing/test_math.cpp
        algorithm/gmm/gmm.cpp
        algorithm/gmm/gmmResult.h
        algorithm/gmm/helpers/sampling.cpp
        algorithm/gmm/helpers/mathematical.cpp
        algorithm/regularization/CovarianceMatrixRegularizer.cpp
        algorithm/regularization/CovarianceMatrixRegularizer.h
        algorithm/regularization/EmpiricalRegularizer.cpp
)
target_link_libraries(gmm_tests PRIVATE GTest::gtest_main GTest::gtest Eigen3::Eigen)

include(GoogleTest)
gtest_discover_tests(gmm_tests)