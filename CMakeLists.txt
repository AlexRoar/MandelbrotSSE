cmake_minimum_required(VERSION 3.17)
project(Mandelbrot)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_library(SDL2_LIBRARY NAME SDL2)
find_package(SDL2 REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(SDL2_image REQUIRED)

include_directories(${SDL2_INCLUDE_DIR})
include_directories(${SDL2_MIXER_INCLUDE_DIRS})
include_directories(${SDL2_IMAGE_INCLUDE_DIRS})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib -lSDL2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -msse4.1 -O3")

add_executable(Mandelbrot main.cpp src/mandutils.h src/Complex.h src/Graphics.h)
include_directories(Mandelbrot ${SDL2_INCLUDE_DIRS})
target_link_libraries(Mandelbrot ${SDL2_LIBRARIES} ${SDL2_MIXER_LIBRARY} ${SDL2_IMAGE_LIBRARY})