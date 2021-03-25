#include <cstdio>
#include <SDL.h>
#include "src/Graphics.h"

const int catPosix = 180;
const int catPosiy = 230;


SDL_Surface* simpleOverlay(SDL_Surface *tableOld, SDL_Surface *cat) {
    SDL_Surface* table = createSurface(tableOld->w, tableOld->h);
    SDL_BlitSurface(tableOld, NULL,table, NULL);
    for (int h = 0; h < cat->h; h++) {
        for (int w = 0; w < cat->w; w++) {
            Uint32 pixelFront = getPixel(cat, w, h);
            Uint32 pixelBack = getPixel(table, w + catPosix, h + catPosiy);
            Uint32 final = 0;
            unsigned char alpha = ((unsigned char*)&pixelFront)[3];
            ((unsigned char*)&final)[3] = ((255 - alpha) + ((unsigned char*)&pixelBack)[3]) / 2;

            for (int i = 0; i < 3; i++){
                ((unsigned char*)&final)[i] = ((255 - alpha) * ((unsigned char*)&pixelBack)[i] + (alpha) * ((unsigned char*)&pixelFront)[i]) / 255;
            }

            setPixel(table, w+catPosix, h+catPosiy, final);
        }
    }
    return table;
}

SDL_Surface* sseOverlay(SDL_Surface *tableOld, SDL_Surface *cat) {
    SDL_Surface* table = createSurface(tableOld->w, tableOld->h);


    const __m128i byte = _mm_set1_epi16(255);
    const __m128i highLoader = _mm_set_epi8(-1, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1, 9, -1, 8);

    SDL_BlitSurface(tableOld, NULL,table, NULL);
    auto effectiveW = cat->w - cat->w % 4;
    for (int h = 0; h < cat->h - 1; h++) {
        for (int w = 0; w < effectiveW; w+=4) {
            Uint32* pixelFront = getPixelPtr(cat, w, h);
            Uint32* pixelBack = getPixelPtr(table, w + catPosix, h + catPosiy);
            Uint32 pixelBackAligned[] = {pixelBack[0], pixelBack[1], pixelBack[2], pixelBack[3]};
            Uint32 pixelFrontAligned[] = {pixelFront[0], pixelFront[1], pixelFront[2], pixelFront[3]};

            __m128i bgLoaded = _mm_load_si128((__m128i*)&pixelBackAligned);
            __m128i frLoaded = _mm_load_si128((__m128i*)&pixelFrontAligned);

            __m128i frLoadedLo = _mm_cvtepu8_epi16(frLoaded);
            __m128i bgLoadedLo = _mm_cvtepu8_epi16(bgLoaded);
            __m128i frLoadedHi = _mm_shuffle_epi8(frLoaded, highLoader);
            __m128i bgLoadedHi = _mm_shuffle_epi8(bgLoaded, highLoader);

            const static __m128i moveA = _mm_set_epi8(-1, 14, -1, 14, -1 , 14, -1, 14,
                                                      -1, 6, -1, 6, -1 , 6, -1, 6);

            __m128i a = _mm_shuffle_epi8(frLoadedLo, moveA);
            __m128i A = _mm_shuffle_epi8(frLoadedHi, moveA);

            frLoadedLo = _mm_mullo_epi16(frLoadedLo, a); // fr * a
            frLoadedHi = _mm_mullo_epi16(frLoadedHi, A);

            bgLoadedLo = _mm_mullo_epi16(bgLoadedLo, _mm_sub_epi16(byte, a)); // bg * (255 - a)
            bgLoadedHi = _mm_mullo_epi16(bgLoadedHi, _mm_sub_epi16(byte, A));

            auto sumLo = _mm_add_epi16(bgLoadedLo, frLoadedLo); // bg * (255 - a) + fr * a
            auto sumHi = _mm_add_epi16(bgLoadedHi, frLoadedHi);

            static const __m128i moveSum = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                        15, 13, 11, 9, 7, 5, 3, 1);

            sumLo = _mm_shuffle_epi8(sumLo, moveSum);
            sumHi = _mm_shuffle_epi8(sumHi, moveSum);

            auto finalColor = (__m128i)(_mm_movelh_ps((__m128)sumLo, (__m128)sumHi));
            _mm_store_si128((__m128i *)(pixelBackAligned), finalColor);
            memcpy(pixelBack, pixelBackAligned, sizeof(pixelBackAligned[0]) * 4);
        }
    }
    return table;
}

int main(){
    auto table = SDL_LoadBMP("assets/Table.bmp");
    auto cat = SDL_LoadBMP("assets/AskhatCat.bmp");

    if (!cat || !table) {
        printf("Unable to open assets!\n");
        return EXIT_FAILURE;
    }
    auto tableAndCat = sseOverlay(table, cat);

    saveSurface(tableAndCat, "catOnTable.png");

    SDL_FreeSurface(tableAndCat);
    SDL_FreeSurface(table);
    SDL_FreeSurface(cat);
}