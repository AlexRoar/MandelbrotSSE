cmake_minimum_required(VERSION 3.17)

project(Mandelbrot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Rpass=loop-vectorize -pthread -msse4.1 -mavx2")
set(CMAKE_CUDA_FLAGS "")

add_executable(MandelbrotTests allTests.cpp src/mandutils.h src/Complex.h src/Graphics.h)
include_directories(MandelbrotTests ${SDL2_INCLUDE_DIRS})
target_link_libraries(MandelbrotTests ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY})

add_executable(MandelbrotWalker walker.cpp src/mandutils.h src/Complex.h src/Graphics.h src/SDLHandlers.h)
include_directories(MandelbrotWalker ${SDL2_INCLUDE_DIRS})
target_link_libraries(MandelbrotWalker ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY})

add_executable(MandelbrotFastest fastest.cpp src/mandutils.h src/Complex.h src/Graphics.h src/SDLHandlers.h)
include_directories(MandelbrotFastest ${SDL2_INCLUDE_DIRS})
target_link_libraries(MandelbrotFastest ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY})

add_executable(CatPlace placeCat.cpp src/mandutils.h src/Complex.h src/Graphics.h src/SDLHandlers.h)
include_directories(CatPlace ${SDL2_INCLUDE_DIRS})
target_link_libraries(CatPlace ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY})