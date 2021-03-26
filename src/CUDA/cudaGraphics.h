//
// Created by alex on 26.03.2021.
//

#ifndef MANDELBROT_CUDAGRAPHICS_H
#define MANDELBROT_CUDAGRAPHICS_H

static void inline setPixel(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    auto *const target_pixel = (Uint32 *) ((Uint8 *) surface->pixels
                                           + y * surface->pitch
                                           + x * surface->format->BytesPerPixel);
    *target_pixel = pixel;
}

static inline Uint32 getPixel(SDL_Surface *surface, int x, int y) {
    return *(Uint32 *) ((Uint8 *) surface->pixels
                        + y * surface->pitch
                        + x * surface->format->BytesPerPixel);
}

static inline Uint32* getPixelPtr(SDL_Surface *surface, int x, int y) {
    return (Uint32 *) ((Uint8 *) surface->pixels
                       + y * surface->pitch
                       + x * surface->format->BytesPerPixel);
}

static void WipeSurface(SDL_Surface *surface) {
    SDL_LockSurface(surface);
    SDL_FillRect(surface, nullptr, SDL_MapRGB(surface->format, 0, 0, 0));
    SDL_UnlockSurface(surface);
}

static SDL_Surface *createSurface(int width, int height) {
    Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
#else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
#endif
    auto surface = SDL_CreateRGBSurface(0, width, height, 32,
                                        rmask,
                                        gmask,
                                        bmask,
                                        amask);
    if (surface == nullptr) {
        SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
        exit(1);
    }
    return surface;
}

static void freeSurface(SDL_Surface *surface) {
    SDL_FreeSurface(surface);
}

static void saveSurface(SDL_Surface *surface, const char *file) {
    IMG_SavePNG(surface, file);
}

__device__ __host__
static inline void setPixelCuda(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    auto *const target_pixel = (Uint32 *) ((Uint8 *) surface->pixels
                                           + y * surface->pitch
                                           + x * surface->format->BytesPerPixel);
    *target_pixel = pixel;
}

__device__ __host__
static inline Uint32 getPixelCuda(SDL_Surface *surface, int x, int y) {
    return *(Uint32 *) ((Uint8 *) surface->pixels
                        + y * surface->pitch
                        + x * surface->format->BytesPerPixel);
}


template <typename c_type>
__device__ unsigned mandelbrotSpeed(const unsigned limitIt, const c_type limitSphere, const Complex<c_type>& c){
    unsigned speed = 0;
    Complex<c_type> zero = {0, 0};
    #pragma unroll 128
    while(speed < limitIt && zero.i + zero.r < limitSphere * limitSphere) {
        zero.square();
        zero.add(c);
        speed++;
    }
    return speed;
}

__host__ __device__
static inline SDL_Surface *getSurfaceFrom(const SDL_Surface *image, void *imageDevicePixels) {
    return SDL_CreateRGBSurfaceFrom(imageDevicePixels,
                                    image->w,
                                    image->h,
                                    32,
                                    image->pitch,
                                    image->format->Rmask,
                                    image->format->Gmask,
                                    image->format->Bmask,
                                    image->format->Amask);
}

template<typename c_type>
__global__ void
mandelbrotProcess(Uint8 *imagePixels,
                  const unsigned pitch,
                  const unsigned bytesPerPixel,
                  const unsigned frameWidth,
                  const unsigned frameHeight,
                  const c_type rePos,
                  const c_type imPos,
                  const c_type sideWidth,
                  const int limit,
                  const dim3 gridInside,
                  const ColorPaletteUF* palette) {

    const unsigned pixelx = blockDim.x * (blockIdx.x * gridInside.x) + threadIdx.x* gridInside.x;
    const unsigned pixely = blockDim.y * (blockIdx.y * gridInside.y) + threadIdx.y* gridInside.y;
    if (pixely >= frameHeight || pixelx >= frameWidth)
        return;
    const c_type imCoefCONST = sideWidth * c_type (frameHeight) / c_type(frameWidth);

#pragma unroll
    for (unsigned h = pixely; h < pixely + gridInside.y && h < frameHeight; h++) {
        const c_type ci = -imPos + imCoefCONST * ( c_type(h) / c_type(frameHeight) - 0.5);
#pragma unroll
        for (unsigned w = pixelx; w < pixelx + gridInside.x && w < frameWidth; w++) {
            Complex<c_type> c = {rePos + sideWidth * ( c_type(w) / frameWidth - 0.5 ), ci};
            Uint32 sp = mandelbrotSpeed<c_type>(limit, 2, c);
            sp = palette->colorNoSmooth(sp, limit);
            *((Uint32 *) ((Uint8 *) imagePixels
                          + h * pitch
                          + w * bytesPerPixel)) = sp;
        }
    }


}

template<typename c_type>
void mandelbrotRender(const ColorPaletteUF& palette, SDL_Surface* image, unsigned frameWidth, unsigned  frameHeight,
                      c_type rePos, c_type imPos, c_type sideWidth, int limit) {
    const size_t image_pixels = frameHeight * frameWidth;
    const size_t image_bytes = image->format->BytesPerPixel * image_pixels;


    void* imageDevicePixels = nullptr;
    auto res = cudaMalloc(&imageDevicePixels, image_bytes);
    if (res != cudaSuccess) {
        printf("Failed to allocate GPU buffer\n");
        return;
    }

    ColorPaletteUF* devicePalette = nullptr;
    cudaMallocManaged((void**) &devicePalette, sizeof (ColorPaletteUF));
    *devicePalette = palette;
    cudaMallocManaged((void **)(&(devicePalette->colors)), ColorPaletteUF::colorsLength * sizeof(Uint32));
    cudaMemcpy(devicePalette->colors, palette.colors, ColorPaletteUF::colorsLength * sizeof(Uint32), cudaMemcpyHostToDevice);

    const dim3 grid(32, 32, 1);
    const dim3 gridInside(2, 2, 1);
    const dim3 gridProcess(frameWidth / (grid.x * gridInside.x) + 1 , frameHeight / (grid.y * gridInside.y) + 1, 1);
//    printf("Launch with (%d %d %d), (%d %d %d)\n", gridProcess.x, gridProcess.y, gridProcess.z,
//           grid.x, grid.y, grid.z);

    mandelbrotProcess<<<gridProcess, grid>>>((Uint8*)imageDevicePixels,
                                             image->pitch,
                                             image->format->BytesPerPixel,
                                             frameWidth,
                                             frameHeight,
                                             rePos, imPos,
                                             sideWidth, limit,
                                             gridInside, devicePalette);
    auto err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

//    cudaDeviceSynchronize();
    err = cudaMemcpy(image->pixels, imageDevicePixels, image_bytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch memcpy (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree((void *) imageDevicePixels);
    cudaFree((void *) devicePalette->colors);
    cudaFree((void *) devicePalette);
}


#endif //MANDELBROT_CUDAGRAPHICS_H
