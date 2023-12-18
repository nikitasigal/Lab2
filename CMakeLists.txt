cmake_minimum_required(VERSION 3.27.0)
project(Lab2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(Lab2 src/main.cpp src/matrix.cpp src/algorithms.cpp src/mpi_util.cpp src/global.cpp)

target_include_directories(Lab2 PUBLIC include)

find_package(MPI REQUIRED)
target_link_libraries(Lab2 PUBLIC MPI::MPI_CXX)
