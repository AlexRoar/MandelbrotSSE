//
// Created by Александр Дремов on 19.03.2021.
//

#ifndef MANDELBROT_MANDUTILS_H
#define MANDELBROT_MANDUTILS_H
#include "Complex.h"
#include <SDL.h>
#include "ColorPalette.h"

unsigned mandelbrotSpeed(unsigned limitIt, double limitSphere, Complex& c){
    unsigned speed = 0;
    Complex zero = {0, 0};
    while(speed < limitIt && zero.abs() < limitSphere ) {
        zero.square();
        zero.add(c);
        speed++;
    }
    c = zero;
    return speed;
}

#endif //MANDELBROT_MANDUTILS_H
