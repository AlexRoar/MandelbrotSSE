//
// Created by Александр Дремов on 19.03.2021.
//

#ifndef MANDELBROT_MANDUTILS_H
#define MANDELBROT_MANDUTILS_H
#include "CUDA/Complex.cuh"
#include <SDL.h>
#include "ColorPalette.h"

#define TIME(code, msg) \
do { \
struct timeval tval_before = {}, tval_after = {}, tval_result = {}; \
bool ended = false;                        \
gettimeofday(&tval_before, NULL);\
{code}                  \
if (!ended){                        \
    gettimeofday(&tval_after, NULL);\
    timersub(&tval_after, &tval_before, &tval_result);                  \
}printf("%s elapsed: %ld.%06ld \n", msg, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);\
} while(0)

#define SET_TIME do { \
gettimeofday(&tval_after, NULL);\
timersub(&tval_after, &tval_before, &tval_result);\
ended = true;\
} while(0)

template <typename T>
unsigned mandelbrotSpeed(unsigned limitIt, double limitSphere, Complex<T>& c){
    unsigned speed = 0;
    Complex<T> zero = {0, 0};
    while(speed < limitIt && zero.abs() < limitSphere ) {
        zero.square();
        zero.add(c);
        speed++;
    }
    c = zero;
    return speed;
}

#endif //MANDELBROT_MANDUTILS_H
