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
    SDL_BlitSurface(tableOld, NULL,table, NULL);
    auto effectiveW = cat->w - cat->w % 4;
    for (int h = 0; h < cat->h; h++) {
        for (int w = 0; w < effectiveW; w+=4) {
            Uint32* pixelFront = getPixelPtr(cat, w, h);
            Uint32* pixelBack = getPixelPtr(table, w + catPosix, h + catPosiy);
            Uint32 final = 0;

            __m128i frLoaded = _mm_load_si128((__m128i*)pixelFront);
            __m128i bgLoaded = _mm_load_si128((__m128i*)pixelBack);

            __m128i frLoadedLo = _mm_cvtepu8_epi16(frLoaded);
            __m128i bgLoadedLo = _mm_cvtepu8_epi16(bgLoaded);
            __m128i frLoadedHi = _mm_shuffle_ps((__m128)frLoaded, (__m128)frLoaded, _MM_SHUFFLE(3,2,3,2));
            __m128i bgLoadedHi = _mm_cvtepu8_epi16(bgLoaded);


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

int main(){
    auto table = SDL_LoadBMP("assets/Table.bmp");
    auto cat = SDL_LoadBMP("assets/AskhatCat.bmp");

    if (!cat || !table) {
        printf("Unable to open assets!\n");
        return EXIT_FAILURE;
    }
    auto tableAndCat = simpleOverlay(table, cat);

    saveSurface(tableAndCat, "catOnTable.png");

    SDL_FreeSurface(tableAndCat);
    SDL_FreeSurface(table);
    SDL_FreeSurface(cat);
}