cmake_minimum_required(VERSION 3.31)
project(EM_regularization)

find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_STANDARD 17)

add_executable(EM_regularization main.cpp
        data.cpp
        algorithm/gmm/gmm.cpp
        algorithm/gmm/gmmResult.h
        algorithm/sampling.cpp
        algorithm/mathematical.cpp
        algorithm/regularization/CovarianceMatrixRegularizer.cpp
        algorithm/regularization/CovarianceMatrixRegularizer.h
        algorithm/regularization/EmpiricalRegularizer.cpp
        algorithm/gmm/gmm.h
        algorithm/Evaluator.cpp)
