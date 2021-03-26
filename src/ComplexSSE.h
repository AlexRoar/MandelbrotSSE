//
// Created by Александр Дремов on 19.03.2021.
// Copyright (c) 2021 Alex Dremov. All rights reserved.
//

#ifndef ComplexSSE_GUARD
#define ComplexSSE_GUARD
#include <cstdlib>
#include <x86intrin.h>

typedef union {
    __m128i v;
    int32_t a[4];
} U32;


void mandelbrotSSEFl(const ColorPaletteUF& palette, SDL_Surface* image, int frameWidth, const int frameHeight, float rePos, float imPos, const float sideWidth, int limit, float r2MaxFloat = 4){
    const int SSE_size = 4;
    frameWidth -= frameWidth % SSE_size;
    imPos *=-1;

    const float imCoefCONST = sideWidth * float(frameHeight) / float(frameWidth);

    const __m128 addVector = _mm_set_ps(0, 1.0, 2.0, 3.0);
    const __m128 FWVector = _mm_set_ps1(float(frameWidth));
    const __m128 FHVector = _mm_set_ps1(float(frameHeight));
    const __m128 half = _mm_set_ps1(float(0.5));
    const __m128 startReal = _mm_set_ps1(rePos);
    const __m128 startIm = _mm_set_ps1(imPos);
    const __m128 sideWidthVec = _mm_set_ps1(sideWidth);
    const __m128 r2Max = _mm_set_ps1(r2MaxFloat);
    const __m128 imCoef = _mm_set_ps1(imCoefCONST);

    float im[SSE_size]  __attribute__((aligned(16))) = {};
    float re[SSE_size]  __attribute__((aligned(16))) = {};
    U32 Nints __attribute__((aligned(16))) = {};

    for (int h = 0; h < frameHeight; h++) {
        __m128 imCoefStored = _mm_set_ps1(float(h));
        imCoefStored = _mm_div_ps(imCoefStored, FHVector); // [h, h, h, h] / frameHeight
        imCoefStored = _mm_sub_ps(imCoefStored, half); // ([h, h, h, h] / frameHeight - 0.5)
        imCoefStored = _mm_mul_ps(imCoefStored, imCoef); // ([h, h, h, h] / frameHeight - 0.5) * imCoef
        imCoefStored = _mm_add_ps(imCoefStored, startIm); // imPos + ([h, h, h, h] / frameHeight - 0.5) * imCoef
        for (int w = 0; w < frameWidth ; w += SSE_size) {
            __m128 reCoefNow = _mm_set_ps1(float(w));

            reCoefNow = _mm_add_ps(reCoefNow, addVector); // [w, w+1, w+2, w+3]
            reCoefNow = _mm_div_ps(reCoefNow, FWVector); // [w, w+1, w+2, w+3] / frameWidth
            reCoefNow = _mm_sub_ps(reCoefNow, half); // ([w, w+1, w+2, w+3] / frameWidth - 0.5)
            reCoefNow = _mm_mul_ps(reCoefNow, sideWidthVec); // ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth
            reCoefNow = _mm_add_ps(reCoefNow, startReal); // rePos + ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth

            __m128i N = _mm_set1_epi32(0);

            __m128 imCoefNow = imCoefStored;
            __m128 reCoefStarted = reCoefNow;
            for (int n = 0; n < limit; n++) {
                __m128 reSq = _mm_mul_ps(reCoefNow, reCoefNow);
                __m128 imSq = _mm_mul_ps(imCoefNow, imCoefNow);
                __m128 r2 = _mm_add_ps(reSq, imSq);

                __m128 cmp = _mm_cmple_ps(r2, r2Max);
                int mask = _mm_movemask_ps(cmp);
                if (!mask) break;

                __m128i addMask = _mm_set_epi32(mask >> 3 & 0x1, mask >> 2 & 0x1, mask >> 1 & 0x1, mask & 0x1);
                N = _mm_add_epi32(N, addMask);
                __m128 xy = _mm_mul_ps(reCoefNow, imCoefNow);
                reCoefNow = _mm_sub_ps(reSq, imSq);
                reCoefNow = _mm_add_ps(reCoefNow, reCoefStarted);
                imCoefNow = _mm_add_ps(xy, xy);
                imCoefNow = _mm_add_ps(imCoefNow, imCoefStored);
            }

            _mm_store_ps(re, imCoefStored);
            _mm_store_ps(im, imCoefStored);
            _mm_store_si128(&Nints.v, N);
            for (int i = 0; i < SSE_size; i ++){
                int speed = Nints.a[SSE_size - i - 1];
                setPixel(image, w + i, h, palette.colorNoSmooth(int(speed), limit));
            }
        }
    }
}

void mandelbrotSSEFlSmooth(const ColorPaletteUF& palette, SDL_Surface* image, int frameWidth, const int frameHeight, float rePos, float imPos, const float sideWidth, int limit, float r2MaxFloat = 4){
    const int SSE_size = 4;
    frameWidth -= frameWidth % SSE_size;
    imPos *=-1;

    const float imCoefCONST = sideWidth * float(frameHeight) / float(frameWidth);

    const __m128 addVector = _mm_set_ps(0, 1.0, 2.0, 3.0);
    const __m128 FWVector = _mm_set_ps1(float(frameWidth));
    const __m128 FHVector = _mm_set_ps1(float(frameHeight));
    const __m128 half = _mm_set_ps1(float(0.5));
    const __m128 startReal = _mm_set_ps1(rePos);
    const __m128 startIm = _mm_set_ps1(imPos);
    const __m128 sideWidthVec = _mm_set_ps1(sideWidth);
    const __m128 r2Max = _mm_set_ps1(r2MaxFloat);
    const __m128 imCoef = _mm_set_ps1(imCoefCONST);

    float im[SSE_size]  __attribute__((aligned(16))) = {};
    float re[SSE_size]  __attribute__((aligned(16))) = {};

    U32 Nints __attribute__((aligned(16))) = {};

    Complex<double> onexit[SSE_size] = {};
    __m128 onExitR = _mm_set_ps1(0);
    __m128 onExitI = _mm_set_ps1(0);
    for (int h = 0; h < frameHeight; h++) {
        __m128 imCoefStored = _mm_set_ps1(float(h));
        imCoefStored = _mm_div_ps(imCoefStored, FHVector); // [h, h, h, h] / frameHeight
        imCoefStored = _mm_sub_ps(imCoefStored, half); // ([h, h, h, h] / frameHeight - 0.5)
        imCoefStored = _mm_mul_ps(imCoefStored, imCoef); // ([h, h, h, h] / frameHeight - 0.5) * imCoef
        imCoefStored = _mm_add_ps(imCoefStored, startIm); // imPos + ([h, h, h, h] / frameHeight - 0.5) * imCoef
        for (int w = 0; w < frameWidth ; w += SSE_size) {
            __m128 reCoefNow = _mm_set_ps1(double (w));

            reCoefNow = _mm_add_ps(reCoefNow, addVector); // [w, w+1, w+2, w+3]
            reCoefNow = _mm_div_ps(reCoefNow, FWVector); // [w, w+1, w+2, w+3] / frameWidth
            reCoefNow = _mm_sub_ps(reCoefNow, half); // ([w, w+1, w+2, w+3] / frameWidth - 0.5)
            reCoefNow = _mm_mul_ps(reCoefNow, sideWidthVec); // ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth
            reCoefNow = _mm_add_ps(reCoefNow, startReal); // rePos + ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth

            __m128i N = _mm_set1_epi32(0);

            __m128 imCoefNow = imCoefStored;
            __m128 reCoefStarted = reCoefNow;
            onexit[0] = {0,0};
            onexit[1] = {0,0};
            onexit[2] = {0,0};
            onexit[3] = {0,0};
            for (int n = 0; n < limit; n++) {
                __m128 reSq = _mm_mul_ps(reCoefNow, reCoefNow);
                __m128 imSq = _mm_mul_ps(imCoefNow, imCoefNow);
                __m128 r2 = _mm_add_ps(reSq, imSq);

                __m128 cmp = _mm_cmple_ps(r2, r2Max);
                int mask = _mm_movemask_ps(cmp);
                _mm_store_ps(re, reCoefNow);
                _mm_store_ps(im, imCoefNow);
                for(int i = 0; i < 4; i++) {
                    char res = int(mask >> i & 0x1);
                    if (res == 0 && onexit[i].i == 0)
                        onexit[i] = {re[i], im[i]};
                }
                if (!mask) break;

                __m128i addMask = _mm_set_epi32(mask >> 3 & 0x1, mask >> 2 & 0x1, mask >> 1 & 0x1, mask & 0x1);
                N = _mm_add_epi32(N, addMask);
                __m128 xy = _mm_mul_ps(reCoefNow, imCoefNow);
                reCoefNow = _mm_sub_ps(reSq, imSq);
                reCoefNow = _mm_add_ps(reCoefNow, reCoefStarted);
                imCoefNow = _mm_add_ps(xy, xy);
                imCoefNow = _mm_add_ps(imCoefNow, imCoefStored);
            }

            _mm_store_ps(re, reCoefNow);
            _mm_store_ps(im, imCoefNow);
            _mm_store_si128(&Nints.v, N);
            for(int i = 0; i < 4; i++) {
                if ( onexit[i].i == 0)
                    onexit[i] = {re[i], im[i]};
            }

            for (int i = 0; i < SSE_size; i ++){
                int speed = Nints.a[SSE_size - i - 1];
                setPixel(image, w + i, h, palette.color(speed, limit, onexit[SSE_size - i - 1]));
            }

        }
    }
}

void  mandelbrotSSEDl(const ColorPaletteUF& palette, SDL_Surface* image, int frameWidth, const int frameHeight, double rePos, double imPos, const double sideWidth, int limit, double r2MaxDouble = 4){
    const int SSE_size = 2;
    frameWidth -= frameWidth % SSE_size;
    imPos *=-1;

    const double imCoefCONST = sideWidth * double(frameHeight) / double(frameWidth);

    const __m128 addVector = _mm_set_pd(0, 1.0);
    const __m128 FWVector = _mm_set_pd1(float(frameWidth));
    const __m128 FHVector = _mm_set_pd1(float(frameHeight));
    const __m128 half = _mm_set_pd1(float(0.5));
    const __m128 startReal = _mm_set_pd1(rePos);
    const __m128 startIm = _mm_set_pd1(imPos);
    const __m128 sideWidthVec = _mm_set_pd1(sideWidth);
    const __m128 r2Max = _mm_set_pd1(r2MaxDouble);
    const __m128 imCoef = _mm_set_pd1(imCoefCONST);

    double im[SSE_size]  __attribute__((aligned(16))) = {};
    double re[SSE_size]  __attribute__((aligned(16))) = {};
    U32 Nints __attribute__((aligned(16))) = {};

    for (int h = 0; h < frameHeight; h++) {
        __m128 imCoefStored = _mm_set_pd1(float(h));
        imCoefStored = _mm_div_pd(imCoefStored, FHVector); // [h, h, h, h] / frameHeight
        imCoefStored = _mm_sub_pd(imCoefStored, half); // ([h, h, h, h] / frameHeight - 0.5)
        imCoefStored = _mm_mul_pd(imCoefStored, imCoef); // ([h, h, h, h] / frameHeight - 0.5) * imCoef
        imCoefStored = _mm_add_pd(imCoefStored, startIm); // imPos + ([h, h, h, h] / frameHeight - 0.5) * imCoef
        for (int w = 0; w < frameWidth ; w += SSE_size) {
            __m128 reCoefNow = _mm_set_pd1(float(w));

            reCoefNow = _mm_add_pd(reCoefNow, addVector); // [w, w+1, w+2, w+3]
            reCoefNow = _mm_div_pd(reCoefNow, FWVector); // [w, w+1, w+2, w+3] / frameWidth
            reCoefNow = _mm_sub_pd(reCoefNow, half); // ([w, w+1, w+2, w+3] / frameWidth - 0.5)
            reCoefNow = _mm_mul_pd(reCoefNow, sideWidthVec); // ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth
            reCoefNow = _mm_add_pd(reCoefNow, startReal); // rePos + ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth

            __m128i N = _mm_set1_epi32(0);

            __m128 imCoefNow = imCoefStored;
            __m128 reCoefStarted = reCoefNow;
            for (int n = 0; n < limit; n++) {
                __m128 reSq = _mm_mul_pd(reCoefNow, reCoefNow);
                __m128 imSq = _mm_mul_pd(imCoefNow, imCoefNow);
                __m128 r2 = _mm_add_pd(reSq, imSq);

                __m128 cmp = _mm_cmple_pd(r2, r2Max);
                int mask = _mm_movemask_pd(cmp);
                if (!mask) break;

                __m128i addMask = _mm_set_epi32(mask >> 3 & 0x1, mask >> 2 & 0x1, mask >> 1 & 0x1, mask & 0x1);
                N = _mm_add_epi32(N, addMask);
                __m128 xy = _mm_mul_pd(reCoefNow, imCoefNow);
                reCoefNow = _mm_sub_pd(reSq, imSq);
                reCoefNow = _mm_add_pd(reCoefNow, reCoefStarted);
                imCoefNow = _mm_add_pd(xy, xy);
                imCoefNow = _mm_add_pd(imCoefNow, imCoefStored);
            }

            _mm_store_pd(re, imCoefStored);
            _mm_store_pd(im, imCoefStored);
            _mm_store_si128(&Nints.v, N);
            for (int i = 0; i < SSE_size; i ++){
                int speed = Nints.a[SSE_size - i - 1];
                setPixel(image, w + i, h, palette.colorNoSmooth(int(speed), limit));
            }

        }
    }
}

void  mandelbrotSSEDlSmooth(const ColorPaletteUF& palette, SDL_Surface* image, int frameWidth, const int frameHeight, double rePos, double imPos, const double sideWidth, int limit, double r2MaxDouble = 4){
    const int SSE_size = 2;
    frameWidth -= frameWidth % SSE_size;
    imPos *=-1;

    const double imCoefCONST = sideWidth * double(frameHeight) / double(frameWidth);

    const __m128 addVector = _mm_set_pd(0, 1.0);
    const __m128 FWVector = _mm_set_pd1(float(frameWidth));
    const __m128 FHVector = _mm_set_pd1(float(frameHeight));
    const __m128 half = _mm_set_pd1(0.5);
    const __m128 startReal = _mm_set_pd1(rePos);
    const __m128 startIm = _mm_set_pd1(imPos);
    const __m128 sideWidthVec = _mm_set_pd1(sideWidth);
    const __m128 r2Max =_mm_set_pd1(r2MaxDouble);
    const __m128 imCoef = _mm_set_pd1(imCoefCONST);

    double im[SSE_size]  __attribute__((aligned(16))) = {};
    double re[SSE_size]  __attribute__((aligned(16))) = {};

    U32 Nints __attribute__((aligned(16))) = {};

    Complex<double> onexit[SSE_size] = {};
    for (int h = 0; h < frameHeight; h++) {
        __m128 imCoefStored = _mm_set_pd1(float(h));
        imCoefStored = _mm_div_pd(imCoefStored, FHVector); // [h, h, h, h] / frameHeight
        imCoefStored = _mm_sub_pd(imCoefStored, half); // ([h, h, h, h] / frameHeight - 0.5)
        imCoefStored = _mm_mul_pd(imCoefStored, imCoef); // ([h, h, h, h] / frameHeight - 0.5) * imCoef
        imCoefStored = _mm_add_pd(imCoefStored, startIm); // imPos + ([h, h, h, h] / frameHeight - 0.5) * imCoef
        for (int w = 0; w < frameWidth ; w += SSE_size) {
            __m128 reCoefNow = _mm_set_pd1(double(w));

            reCoefNow = _mm_add_pd(reCoefNow, addVector); // [w, w+1, w+2, w+3]
            reCoefNow = _mm_div_pd(reCoefNow, FWVector); // [w, w+1, w+2, w+3] / frameWidth
            reCoefNow = _mm_sub_pd(reCoefNow, half); // ([w, w+1, w+2, w+3] / frameWidth - 0.5)
            reCoefNow = _mm_mul_pd(reCoefNow, sideWidthVec); // ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth
            reCoefNow = _mm_add_pd(reCoefNow, startReal); // rePos + ([w, w+1, w+2, w+3] / frameWidth - 0.5) * sideWidth

            __m128i N = _mm_set1_epi32(0);

            __m128 imCoefNow = imCoefStored;
            __m128 reCoefStarted = reCoefNow;
            onexit[0] = {0,0};
            onexit[1] = {0,0};
            for (int n = 0; n < limit; n++) {
                __m128 reSq = _mm_mul_pd(reCoefNow, reCoefNow);
                __m128 imSq = _mm_mul_pd(imCoefNow, imCoefNow);
                __m128 r2 = _mm_add_pd(reSq, imSq);

                __m128 cmp = _mm_cmple_pd(r2, r2Max);
                int mask = _mm_movemask_pd(cmp);
                _mm_store_pd(re, reCoefNow);
                _mm_store_pd(im, imCoefNow);
                for(int i = 0; i < 4; i++) {
                    char res = int(mask >> i & 0x1);
                    if (res == 0 && onexit[i].i == 0)
                        onexit[i] = {re[i], im[i]};
                }
                if (!mask) break;

                __m128i addMask = _mm_set_epi32(mask >> 3 & 0x1, mask >> 2 & 0x1, mask >> 1 & 0x1, mask & 0x1);
                N = _mm_add_epi32(N, addMask);
                __m128 xy = _mm_mul_pd(reCoefNow, imCoefNow);
                reCoefNow = _mm_sub_pd(reSq, imSq);
                reCoefNow = _mm_add_pd(reCoefNow, reCoefStarted);
                imCoefNow = _mm_add_pd(xy, xy);
                imCoefNow = _mm_add_pd(imCoefNow, imCoefStored);
            }

            _mm_store_pd(re, reCoefNow);
            _mm_store_pd(im, imCoefNow);
            _mm_store_si128(&Nints.v, N);
            for(int i = 0; i < 4; i++) {
                if ( onexit[i].i == 0)
                    onexit[i] = {re[i], im[i]};
            }

            for (int i = 0; i < SSE_size; i ++){
                int speed = Nints.a[SSE_size - i - 1];
                setPixel(image, w + i, h, palette.color(speed, limit, onexit[SSE_size - i - 1]));
            }

        }
    }
}

#endif //ComplexSSE_GUARD
