//
// Created by Александр Дремов on 20.03.2021.
//

#ifndef MANDELBROT_SDLHANDLERS_H
#define MANDELBROT_SDLHANDLERS_H
#include <SDL.h>
#include "CUDA/libraryMandelbrotCXXAPI.h"
#include "Graphics.h"
#include "ComplexSSE.h"

struct App {
    bool smooth = false;
    int fastestModeDs = 2;
    int fastestModePs = 2;
    double switchWidth = 9.35799e-05;
    SDL_Window *pWindow;
    SDL_Renderer *pRenderer;
    int renderType = 6;
    double scalingSpeed = 1.05;
    double moveSpeed = 50;
    double limitSpeed = 1.15;
    bool antiAlias = false;
    int frameWidth = 640 * 2;
    int frameHeight = 420 * 2;
    int pNum = 0;
    SDL_DisplayMode dm;
    ColorPaletteUF palette = {};
    SDL_Surface *image = nullptr;
    double rePos = 0;
    double imPos = 0;
    double sideWidth = 2;
    int limitIter = 256;
};
App thisApp;

void rerender(SDL_Window *pWindow, SDL_Renderer *pRenderer, SDL_Surface *image) {
    if (thisApp.smooth){
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

            case 2: {
                printf("CUDA float render type\n");
                mandelbrotRender<float>(thisApp.palette, image,
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
    } else {
        switch (thisApp.renderType) {
            case 0: {
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
            case 1: {
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
            case 2: {
                printf("CUDA double render type\n");
                mandelbrotRender<double>(thisApp.palette, image,
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
    }

    if (thisApp.antiAlias)
        antialiasImage(image, 1, 0.2);
}

void hadlerUp(SDL_Event &e) {
    thisApp.imPos += thisApp.sideWidth / thisApp.moveSpeed;
}

void hadlerDown(SDL_Event &e) {
    thisApp.imPos -= thisApp.sideWidth / thisApp.moveSpeed;
}

void hadlerLeft(SDL_Event &e) {
    thisApp.rePos -= thisApp.sideWidth / thisApp.moveSpeed;
}

void hadlerRight(SDL_Event &e) {
    thisApp.rePos += thisApp.sideWidth / thisApp.moveSpeed;
}

void hadlerZ(SDL_Event &e) {
    thisApp.sideWidth /= thisApp.scalingSpeed;
    if (thisApp.sideWidth < thisApp.switchWidth)
        thisApp.renderType = thisApp.fastestModeDs;
    else
        thisApp.renderType = thisApp.fastestModePs;
}

void hadlerX(SDL_Event &e) {
    thisApp.sideWidth *= thisApp.scalingSpeed;
    if (thisApp.sideWidth < thisApp.switchWidth)
        thisApp.renderType = thisApp.fastestModeDs;
    else
        thisApp.renderType = thisApp.fastestModePs;
}

void hadlerC(SDL_Event &e) {
    saveSurface(thisApp.image, "snapdot.png");
}

void hadlerR(SDL_Event &e) {
    thisApp.smooth = !thisApp.smooth;
}

void hadlerW(SDL_Event &e) {
    thisApp.limitIter = double(thisApp.limitIter) * double(thisApp.limitSpeed);
}

void hadlerS(SDL_Event &e) {
    thisApp.limitIter = double(thisApp.limitIter) / double(thisApp.limitSpeed);
}

void hadlerQ(SDL_Event &e) {
    thisApp.antiAlias = !thisApp.antiAlias;
}

void hadlerE(SDL_Event &e) {
    thisApp.renderType++;
}

void hadlerA(SDL_Event &e) {
    thisApp.pNum = (thisApp.pNum + 1) % ColorPaletteUF::pNumMax;
    thisApp.palette.dest();
    thisApp.palette.init(thisApp.pNum);
}

void hadlerF(SDL_Event &e) {
    SDL_RestoreWindow(thisApp.pWindow);
    SDL_SetWindowSize(thisApp.pWindow, thisApp.dm.w, thisApp.dm.h);
    thisApp.frameWidth = thisApp.dm.w;
    thisApp.frameHeight = thisApp.dm.h;
    SDL_SetWindowPosition(thisApp.pWindow, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_SetWindowFullscreen(thisApp.pWindow, SDL_WINDOW_FULLSCREEN_DESKTOP);
    thisApp.image = SDL_GetWindowSurface(thisApp.pWindow);
}

void hadlerT(SDL_Event &e) {
    int lastW = thisApp.frameWidth, lastH = thisApp.frameHeight;
    thisApp.frameWidth = 15360;
    thisApp.frameHeight = 8640;
    SDL_Surface* high = createSurface(thisApp.frameWidth, thisApp.frameHeight);
    rerender(thisApp.pWindow, thisApp.pRenderer,high);
    saveSurface(high, "highres.png");
    SDL_FreeSurface(high);
    thisApp.frameWidth = lastW;
    thisApp.frameHeight = lastH;
}

struct KeyboardMap {
    SDL_Scancode code;

    void (*handler)(SDL_Event &e);
};

constexpr KeyboardMap keyboardHandlers[] = {
        {SDL_SCANCODE_UP,    hadlerUp},
        {SDL_SCANCODE_DOWN,  hadlerDown},
        {SDL_SCANCODE_LEFT,  hadlerLeft},
        {SDL_SCANCODE_RIGHT, hadlerRight},
        {SDL_SCANCODE_Z,     hadlerZ},
        {SDL_SCANCODE_X,     hadlerX},
        {SDL_SCANCODE_W,     hadlerW},
        {SDL_SCANCODE_S,     hadlerS},
        {SDL_SCANCODE_A,     hadlerA},
        {SDL_SCANCODE_Q,     hadlerQ},
        {SDL_SCANCODE_E,     hadlerE},
        {SDL_SCANCODE_F,     hadlerF},
        {SDL_SCANCODE_C,     hadlerC},
        {SDL_SCANCODE_R,     hadlerR},
        {SDL_SCANCODE_T,     hadlerT},
};
#endif //MANDELBROT_SDLHANDLERS_H
