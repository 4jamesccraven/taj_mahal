cmake_minimum_required(VERSION 3.30)
project(taj_mahal)

SET(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)

pybind11_add_module(taj_mahal src/taj_mahal.cpp)

target_link_libraries(taj_mahal PRIVATE Eigen3::Eigen Boost::boost)
