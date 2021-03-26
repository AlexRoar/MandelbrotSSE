cmake_minimum_required(VERSION 3.15)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS_DEBUG "-arch=sm_30 -O0 -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-arch=sm_30 -O9")
set(CMAKE_CUDA_HOST_COMPILER clang)

project(Mandelbrot LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_library(CUDAHelpers  src/CUDA/Complex.cu src/CUDA/Complex.cuh src/CUDA/libraryMandelbrot.cu)
target_compile_features(CUDAHelpers PUBLIC cxx_std_14)
set_target_properties( CUDAHelpers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
include_directories(CUDAHelpers "/usr/local/cuda/samples/common/inc/")
include_directories(CUDAHelpers "/usr/local/cuda/cuda-toolkit/include")

find_library(SDL2_LIBRARY NAME SDL2)
find_package(SDL2 REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(SDL2_image REQUIRED)

include_directories(${SDL2_INCLUDE_DIR})
include_directories(${SDL2_MIXER_INCLUDE_DIRS})
include_directories(${SDL2_IMAGE_INCLUDE_DIRS})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/lib -lSDL2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Rpass=loop-vectorize -pthread -msse4.1 -mavx")

add_executable(MandelbrotTests allTests.cpp src/ColorPalette.cpp src/mandutils.h src/CUDA/Complex.cuh src/Graphics.h)
include_directories(MandelbrotTests ${SDL2_INCLUDE_DIRS})
target_link_libraries(MandelbrotTests ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY} CUDAHelpers)

add_executable(MandelbrotWalker walker.cpp src/ColorPalette.cpp src/mandutils.h src/CUDA/Complex.cuh  src/Graphics.h src/SDLHandlers.h)
include_directories(MandelbrotWalker ${SDL2_INCLUDE_DIRS})
target_link_libraries(MandelbrotWalker ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY} CUDAHelpers)

add_executable(MandelbrotFastest fastest.cpp src/ColorPalette.cpp src/mandutils.h src/CUDA/Complex.cuh  src/Graphics.h src/SDLHandlers.h)
include_directories(MandelbrotFastest ${SDL2_INCLUDE_DIRS})
target_link_libraries(MandelbrotFastest ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY} CUDAHelpers)

add_executable(CatPlace placeCat.cpp src/mandutils.h src/ColorPalette.cpp src/CUDA/Complex.cuh  src/Graphics.h src/SDLHandlers.h)
include_directories(CatPlace ${SDL2_INCLUDE_DIRS})
target_link_libraries(CatPlace ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY} CUDAHelpers)
link_libraries(CUDAHelpers)