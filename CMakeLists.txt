cmake_minimum_required(VERSION 3.31)
project(EM_regularization)

set(CMAKE_CXX_STANDARD 20)

add_executable(EM_regularization main.cpp
        data.h
        data.cpp
        algorithm/gmm.cpp)
