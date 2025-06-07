cmake_minimum_required(VERSION 3.31)
project(EM_regularization)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DEIGEN_USE_BLAS)

find_package(Eigen3 3.4 REQUIRED)
find_package(BLAS REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
add_executable(EM_regularization src/main.cpp
        src/data.cpp
        src/algorithm/gmm/gmm.cpp
        src/algorithm/gmm/gmmResult.h
        src/algorithm/gmm/helpers/mathematical.cpp
        src/algorithm/regularization/CovarianceMatrixRegularizer.cpp
        src/algorithm/regularization/EmpiricalRegularizer.cpp
        src/algorithm/GeneticalAlgorithm.cpp
        src/external/hungarian/hungarian.cpp
        src/algorithm/SqMahalanobis.cpp
)

enable_testing()
find_package(GTest REQUIRED)

add_executable(gmm_tests
        src/data.cpp
        Testing/test_regularization.cpp
        Testing/test_gmm.cpp
        Testing/test_math.cpp
        Testing/test_GA.cpp
        src/algorithm/gmm/gmm.cpp
        src/algorithm/gmm/gmmResult.h
        src/algorithm/gmm/helpers/mathematical.cpp
        src/algorithm/regularization/CovarianceMatrixRegularizer.cpp
        src/algorithm/regularization/EmpiricalRegularizer.cpp
        src/algorithm/GeneticalAlgorithm.cpp
        src/external/hungarian/hungarian.cpp
        src/algorithm/SqMahalanobis.cpp
)
target_link_libraries(gmm_tests PRIVATE GTest::gtest_main GTest::gtest Eigen3::Eigen ${BLAS_LIBRARIES})
target_link_libraries(EM_regularization PRIVATE ${BLAS_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(gmm_tests)
