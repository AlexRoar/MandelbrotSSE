//
// Created by Александр Дремов on 20.03.2021.
//

#ifndef MANDELBROT_SDLHANDLERS_H
#define MANDELBROT_SDLHANDLERS_H
#include <SDL.h>

struct App {
    int fastestModeDs = 4;
    int fastestModePs = 7;
    double switchWidth = 9.35799e-05;
    SDL_Window *pWindow;
    SDL_Renderer *pRenderer;
    int renderType = 6;
    double scalingSpeed = 1.05;
    double moveSpeed = 50;
    double limitSpeed = 1.15;
    bool antiAlias = false;
    int frameWidth = 1920;
    int frameHeight = 1080;
    int pNum = 0;
    SDL_DisplayMode dm;
    ColorPaletteUF palette = {};
    SDL_Surface *image = nullptr;
    double rePos = -0.4;
    double imPos = 0.6;
    double sideWidth = 2;
    int limitIter = 250;
};
App thisApp;
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
};
#endif //MANDELBROT_SDLHANDLERS_H
