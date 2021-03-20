#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include <sys/time.h>
#include <cstdio>
#include <SDL.h>

struct App {
    constexpr static int frameWidth = 1920 / 2;
    constexpr static int frameHeight = 1080 / 2;
    int pNum = 0;
    ColorPaletteUF palette = {};
    SDL_Surface *image = nullptr;
    double rePos = 0;
    double imPos = 0;
    double sideWidth = 2;
    int limitIter = 250;
};

App thisApp;

bool init(SDL_Window*& pWindow, SDL_Renderer*& pRenderer);
void close(SDL_Window *pWindow, SDL_Renderer *pRenderer);
void PrintKeyInfo( SDL_KeyboardEvent *key );

void hadlerUp(SDL_Event& e){
    thisApp.imPos += thisApp.sideWidth / 100;
}
void hadlerDown(SDL_Event& e){
    thisApp.imPos -= thisApp.sideWidth / 100;
}
void hadlerLeft(SDL_Event& e){
    thisApp.rePos -= thisApp.sideWidth / 100;
}
void hadlerRight(SDL_Event& e){
    thisApp.rePos += thisApp.sideWidth / 100;
}
void hadlerLshift(SDL_Event& e){
    thisApp.sideWidth /= 1.2;
}
void hadlerSpace(SDL_Event& e){
    thisApp.sideWidth *=  1.2;
}
void hadlerW(SDL_Event& e){
    thisApp.limitIter *=  2;
}
void hadlerS(SDL_Event& e){
    thisApp.limitIter /=  2;
}

void hadlerA(SDL_Event& e){
    thisApp.pNum = (thisApp.pNum + 1) % ColorPaletteUF::pNumMax;
    thisApp.palette.dest();
    thisApp.palette.init(thisApp.pNum);
}

struct KeyboardMap {
    SDL_Scancode code;
    void (*handler)(SDL_Event& e);
};

constexpr KeyboardMap keyboardHandlers[] = {
        {SDL_SCANCODE_UP, hadlerUp},
        {SDL_SCANCODE_DOWN, hadlerDown},
        {SDL_SCANCODE_LEFT, hadlerLeft},
        {SDL_SCANCODE_RIGHT, hadlerRight},
        {SDL_SCANCODE_LSHIFT, hadlerLshift},
        {SDL_SCANCODE_SPACE, hadlerSpace},
        {SDL_SCANCODE_W, hadlerW},
        {SDL_SCANCODE_S, hadlerS},
        {SDL_SCANCODE_A, hadlerA},
};

void rerender(SDL_Window *pWindow, SDL_Renderer *pRenderer, SDL_Surface* image) {
    mandelbrotVectored<double8, long8, double, 8>(thisApp.palette,
                                                  image,
                                                  App::frameWidth,
                                                  App::frameHeight,
                                                  thisApp.rePos,
                                                  thisApp.imPos,
                                                  thisApp.sideWidth,
                                                  thisApp.limitIter);
    antialiasImage(image, 1, 0.2);
}

int main(){
    SDL_Window *pWindow = nullptr;
    SDL_Renderer *pRenderer = nullptr;
    init(pWindow, pRenderer);

    thisApp.image = SDL_GetWindowSurface(pWindow);
    thisApp.palette.init(thisApp.pNum);

    rerender(pWindow, pRenderer, thisApp.image);

    bool quit = false;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            switch (e.type) {
                case SDL_QUIT: {
                    quit = true;
                    break;
                }
                case SDL_KEYDOWN: {
                    bool changed = false;
                    for(int i = 0; i < sizeof(keyboardHandlers) / sizeof(KeyboardMap); i++) {
                        if (keyboardHandlers[i].code == e.key.keysym.scancode){
                            keyboardHandlers[i].handler(e);
                            changed = true;
                        }
                    }
                    if (changed){
                        rerender(pWindow, pRenderer, thisApp.image);
                        SDL_UpdateWindowSurface(pWindow);
                    }
                    PrintKeyInfo( &e.key );
                }
            }
        }
    }

    close(pWindow, pRenderer);
}

void PrintKeyInfo( SDL_KeyboardEvent *key ){
    /* Is it a release or a press? */
    if( key->type == SDL_KEYUP )
        printf( "Release:- " );
    else
        printf( "Press:- " );

    /* Print the hardware scancode first */
    printf( "Scancode: 0x%02X", key->keysym.scancode );
    /* Print the name of the key */
    printf( ", Name: %s", SDL_GetKeyName( key->keysym.sym ) );
    printf( "\n" );
}


bool init(SDL_Window*& pWindow, SDL_Renderer*& pRenderer) {
    SDL_Init(SDL_INIT_VIDEO);
    pWindow = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, thisApp.frameWidth, thisApp.frameHeight, 0);
//    SDL_CreateWindowAndRenderer(thisApp.frameWidth, thisApp.frameHeight, 0, &pWindow, &pRenderer);
//    SDL_SetRenderDrawColor(pRenderer, 0, 0, 0, 255);
//    SDL_RenderClear(pRenderer);
//    SDL_RenderPresent(pRenderer);
    return true;
}

void close(SDL_Window *pWindow, SDL_Renderer *pRenderer) {
    SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    SDL_Quit();
}