#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include "src/SDLHandlers.h"
#include <cstdio>
#include <SDL.h>


bool init(SDL_Window *&pWindow, SDL_Renderer *&pRenderer);

void close(SDL_Window *pWindow, SDL_Renderer *pRenderer);

void PrintKeyInfo(SDL_KeyboardEvent *key);


int main() {
    SDL_Window *pWindow = nullptr;
    SDL_Renderer *pRenderer = nullptr;
    init(pWindow, pRenderer);

    thisApp.pWindow = pWindow;
    thisApp.pWindow = pWindow;
    thisApp.image = SDL_GetWindowSurface(pWindow);
    thisApp.palette.init(thisApp.pNum);

    rerender(pWindow, pRenderer, thisApp.image);
    SDL_UpdateWindowSurface(pWindow);

    if (SDL_GetDesktopDisplayMode(0, &thisApp.dm)) {
        printf("Error getting desktop display mode\n");
        return EXIT_FAILURE;
    }

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
                    for (int i = 0; i < sizeof(keyboardHandlers) / sizeof(KeyboardMap); i++) {
                        if (keyboardHandlers[i].code == e.key.keysym.scancode) {
                            keyboardHandlers[i].handler(e);
                            changed = true;
                        }
                    }
                    if (changed) {
                        printf("Positioning... R(%.20lg) I(%.20lg) W(%.20lg) It(%d)\n",
                               thisApp.rePos, thisApp.imPos,
                               thisApp.sideWidth, thisApp.limitIter);
                        rerender(pWindow, pRenderer, thisApp.image);
                        SDL_UpdateWindowSurface(pWindow);
                        SDL_FlushEvent(SDL_KEYDOWN);
                    }
                }
            }
        }
    }

    close(pWindow, pRenderer);
}

void PrintKeyInfo(SDL_KeyboardEvent *key) {
    /* Is it a release or a press? */
    if (key->type == SDL_KEYUP)
        printf("Release:- ");
    else
        printf("Press:- ");

    /* Print the hardware scancode first */
    printf("Scancode: 0x%02X", key->keysym.scancode);
    /* Print the name of the key */
    printf(", Name: %s", SDL_GetKeyName(key->keysym.sym));
    printf("\n");
}


bool init(SDL_Window *&pWindow, SDL_Renderer *&pRenderer) {
    SDL_Init(SDL_INIT_VIDEO);
    pWindow = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, thisApp.frameWidth,
                               thisApp.frameHeight, 0);
//    SDL_CreateWindowAndRenderer(thisApp.frameWidth, thisApp.frameHeight, 0, &pWindow, &pRenderer);
    SDL_SetWindowFullscreen(pWindow, 0);
    SDL_SetRenderDrawColor(pRenderer, 0, 0, 0, 255);
    SDL_RenderClear(pRenderer);
    return true;
}

void close(SDL_Window *pWindow, SDL_Renderer *pRenderer) {
    SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    SDL_Quit();
}