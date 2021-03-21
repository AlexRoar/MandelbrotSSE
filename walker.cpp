#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include "src/SDLHandlers.h"
#include <cstdio>
#include <SDL.h>


bool init(SDL_Window *&pWindow, SDL_Renderer *&pRenderer);

void close(SDL_Window *pWindow, SDL_Renderer *pRenderer);

void PrintKeyInfo(SDL_KeyboardEvent *key);


void rerender(SDL_Window *pWindow, SDL_Renderer *pRenderer, SDL_Surface *image) {
    switch (thisApp.renderType) {
        case 0: {
            printf("SSE double smooth render type\n");
            mandelbrotSSEDlSmooth(thisApp.palette,
                                  image,
                                  thisApp.frameWidth,
                                  thisApp.frameHeight,
                                  thisApp.rePos,
                                  thisApp.imPos,
                                  thisApp.sideWidth,
                                  thisApp.limitIter);
            break;
        }
        case 1: {
            printf("SSE double render type\n");
            mandelbrotSSEDl(thisApp.palette,
                            image,
                            thisApp.frameWidth,
                            thisApp.frameHeight,
                            thisApp.rePos,
                            thisApp.imPos,
                            thisApp.sideWidth,
                            thisApp.limitIter);
            break;
        }
        case 2: {
            printf("SSE float smooth render type\n");
            mandelbrotSSEFlSmooth(thisApp.palette,
                                  image,
                                  thisApp.frameWidth,
                                  thisApp.frameHeight,
                                  thisApp.rePos,
                                  thisApp.imPos,
                                  thisApp.sideWidth,
                                  thisApp.limitIter);
            break;
        }
        case 3: {
            printf("SSE float render type\n");
            mandelbrotSSEFl(thisApp.palette,
                            image,
                            thisApp.frameWidth,
                            thisApp.frameHeight,
                            thisApp.rePos,
                            thisApp.imPos,
                            thisApp.sideWidth,
                            thisApp.limitIter);
            break;
        }
        case 4: {
            printf("OpenCL double8 render type (fastest)\n");
            mandelbrotVectored<double8, long8, double, 8>(thisApp.palette,
                                                          image,
                                                          thisApp.frameWidth,
                                                          thisApp.frameHeight,
                                                          thisApp.rePos,
                                                          thisApp.imPos,
                                                          thisApp.sideWidth,
                                                          thisApp.limitIter);
            break;
        }
        case 5: {
            printf("OpenCL double16 render type\n");
            mandelbrotVectored<double16, long16, double, 16>(thisApp.palette,
                                                             image,
                                                             thisApp.frameWidth,
                                                             thisApp.frameHeight,
                                                             thisApp.rePos,
                                                             thisApp.imPos,
                                                             thisApp.sideWidth,
                                                             thisApp.limitIter);
            break;
        }
        case 6: {
            printf("OpenCL float16 render type\n");
            mandelbrotVectored<float16, int16, float, 16>(thisApp.palette,
                                                          image,
                                                          thisApp.frameWidth,
                                                          thisApp.frameHeight,
                                                          thisApp.rePos,
                                                          thisApp.imPos,
                                                          thisApp.sideWidth,
                                                          thisApp.limitIter);
            break;
        }
        case 7: {
            printf("OpenCL float32 render type (fastest)\n");
            mandelbrotVectored<float32, int32, float, 32>(thisApp.palette,
                                                          image,
                                                          thisApp.frameWidth,
                                                          thisApp.frameHeight,
                                                          thisApp.rePos,
                                                          thisApp.imPos,
                                                          thisApp.sideWidth,
                                                          thisApp.limitIter);
            break;
        }
        default: {
            thisApp.renderType = 0;
            rerender(pWindow, pRenderer, image);
            return;
        }
    }

    if (thisApp.antiAlias)
        antialiasImage(image, 1, 0.2);
}

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