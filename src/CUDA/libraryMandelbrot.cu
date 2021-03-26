#include <cstdio>
#include "Complex.cuh"
#include "../ColorPalette.h"
#include "cudaGraphics.h"


void testCUDA(){
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("Max Thread Dimensions: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Block Dimensions: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    ColorPaletteUF palette = {};
    palette.init(0);
    auto* surf = createSurface(1920, 1080);
    mandelbrotRender<double>(palette, surf, surf->w, surf->h, -.6,0.3,1,500);
    saveSurface(surf, "cudaHi.png");
}

template void mandelbrotRender<float>(const ColorPaletteUF& palette, SDL_Surface* image, unsigned frameWidth, unsigned  frameHeight,
                                                  float rePos, float imPos, float sideWidth, int limit);