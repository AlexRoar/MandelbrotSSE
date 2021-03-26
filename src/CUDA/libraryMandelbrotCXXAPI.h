//
// Created by alex on 22.03.2021.
//

#ifndef MANDELBROT_LIBRARYMANDELBROTCXXAPI_H
#define MANDELBROT_LIBRARYMANDELBROTCXXAPI_H

template<typename c_type>
void mandelbrotRender(const ColorPaletteUF& palette, SDL_Surface* image, unsigned frameWidth, unsigned  frameHeight,
                      c_type rePos, c_type imPos, c_type sideWidth, int limit);

#endif //MANDELBROT_LIBRARYMANDELBROTCXXAPI_H
