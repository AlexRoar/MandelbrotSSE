//
// Created by Александр Дремов on 19.03.2021.
//

#ifndef MANDELBROT_GRAPHICS_H
#define MANDELBROT_GRAPHICS_H
#include <SDL.h>
#include <SDL_image.h>

typedef float float512 __attribute__((ext_vector_type(512)));
typedef float float32 __attribute__((ext_vector_type(32)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef int int512 __attribute__((ext_vector_type(512)));
typedef int int32 __attribute__((ext_vector_type(32)));
typedef int int16 __attribute__((ext_vector_type(16)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef long long32 __attribute__((ext_vector_type(32)));
typedef long long16 __attribute__((ext_vector_type(16)));
typedef long long8 __attribute__((ext_vector_type(8)));
typedef long long4 __attribute__((ext_vector_type(4)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef double double8 __attribute__((ext_vector_type(8)));
typedef double double16 __attribute__((ext_vector_type(16)));
typedef double double32 __attribute__((ext_vector_type(32)));

void inline setPixel(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    auto *const target_pixel = (Uint32 *) ((Uint8 *) surface->pixels
                                           + y * surface->pitch
                                           + x * surface->format->BytesPerPixel);
    *target_pixel = pixel;
}
Uint32 getPixel(SDL_Surface *surface, int x, int y) {
    return *(Uint32 *) ((Uint8 *) surface->pixels
                                           + y * surface->pitch
                                           + x * surface->format->BytesPerPixel);
}

void WipeSurface(SDL_Surface *surface) {
    SDL_LockSurface(surface);
    SDL_FillRect(surface, nullptr, SDL_MapRGB(surface->format, 0, 0, 0));
    SDL_UnlockSurface(surface);
}

SDL_Surface *createSurface(int width, int height) {
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

void freeSurface(SDL_Surface *surface) {
    SDL_FreeSurface(surface);
}

void saveSurface(SDL_Surface *surface, const char *file) {
    IMG_SavePNG(surface, file);
}

SDL_Surface *showPalette(int pNo) {
    ColorPaletteUF palette = {};
    palette.init(pNo);
    const int height = 64;
    auto surface = createSurface(ColorPaletteUF::colorsLength + 1, height);
    for (int i = 0; i < ColorPaletteUF::colorsLength; i++) {
        for (int j = 0; j < height; j++)
            setPixel(surface, i % surface->w, (i / surface->w) * height + j, palette.colors[i]);
    }
    palette.dest();
    return surface;
}

Uint32 surfaceSample(SDL_Surface *surface, int x, int y, unsigned sampleArea, double percent){
    unsigned colors[4] = {0, 0, 0, 0};
    double c = 0;
    for (int w = x - int(sampleArea); w < x + int(sampleArea) && w < surface->w; w++){
        if (w < 0)
            continue;
        for (int h = y - int(sampleArea); h < y + int(sampleArea) && h < surface->h; h++){
            if (h < 0)
                continue;
            const Uint32 newP = getPixel(surface, w, h);
            auto* colorsMapNew = (unsigned char*)(&newP);
            if (x == w && y == h){
                c += int(1.0 / percent);
            } else {
                c++;
            }
            for (int i = 0; i < 4; i++) {
                unsigned color = colorsMapNew[i];
                if (x == w && y == h){
                    color *= int(1.0 / percent);
                }
                colors[i] += color;
            }
        }
    }

    for (unsigned int & color : colors)
        color = unsigned(double(color) / (c));
    Uint32 res = 0;
    auto* colorsMapNew = (unsigned char*)(&res);
    for (int i = 0; i < 4; i++)
        colorsMapNew[i] = (unsigned char)colors[i];
    return res;
}

void antialiasImage(SDL_Surface *surface, unsigned sampleArea, double percent){
    for (int w = 0; w < surface->w; w++){
        for (int h = 0; h < surface->h; h++){
            setPixel(surface, w, h, surfaceSample(surface, w, h, sampleArea, percent));
        }
    }
}

void mandelbrotSimple(const ColorPaletteUF& palette, SDL_Surface* image, int frameWidth, int frameHeight, double rePos, double imPos, double sideWidth,
                 int limit) {
    const double imCoefCONST = sideWidth * double (frameHeight) / double(frameWidth);
    for (int h = 0; h < frameHeight; h++) {
        const double ci = -imPos + imCoefCONST * ( double(h) / double(frameHeight) - 0.5);
        for (int w = 0; w < frameWidth; w++) {
            Complex c = {rePos + sideWidth * ( double(w) / double(frameWidth) - 0.5 ),
                         ci};
            unsigned speed = 0;
            Complex zero = {0, 0};
            while(speed < limit && zero.absNoSqrt() < 4 ) {
                zero.square();
                zero.add(c);
                speed++;
            }
            c = zero;
//            printf("%d\n", int(i));
            setPixel(image, w, h, palette.color(speed,limit, c));
        }
    }
}

template<typename data, typename counter, typename generic, int dSize>
void mandelbrotVectored(const ColorPaletteUF& palette, SDL_Surface* image, int frameWidth, int frameHeight, generic rePos, generic imPos, generic sideWidth,
                      int limit) {
    frameWidth -= frameWidth % dSize;
    imPos *= -1;
    const data _imCoefCONST = sideWidth * frameHeight / frameWidth;

    const generic r2Max = 4;
    data _adder = {};
    for(int i = 0; i < dSize; i++) {
        _adder[i] = generic(i);
    }


    for (int h = 0; h < frameHeight; h++) {
        const data _ciAll = imPos + _imCoefCONST * (data(h) / frameHeight - data(0.5));
        for (int w = 0; w < frameWidth; w += dSize) {
            data _w = w + _adder;
            data _cr = data(rePos + sideWidth * ( _w / frameWidth - data(0.5)));
            data _ci = _ciAll;
            counter speed(0);
            data _zeror(_cr);
            data _zeroi(_ciAll);

            for (int allSpeed = 0; allSpeed < limit; allSpeed++) {
                data _zeror2 = _zeror * _zeror;
                data _zeroi2 = _zeroi * _zeroi;

                counter r2Cmp = (_zeror2 + _zeroi2) < r2Max;

                speed  += -r2Cmp;
                _zeroi = 2 * _zeror * _zeroi + _ci;
                _zeror = _zeror2 - _zeroi2 + _cr;
            }
            #pragma unroll
            for (int i = 0; i < dSize; i++) {
                setPixel(image, w + i, h, palette.colorNoSmooth(speed[i],limit));
            }
        }
    }
}

#endif //MANDELBROT_GRAPHICS_H