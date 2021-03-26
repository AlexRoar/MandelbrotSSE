#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include <sys/time.h>


constexpr int frameWidth = 1920 * 4;
constexpr int frameHeight = 1080 * 4;

constexpr double rePos = -0.027844723880907473745;
constexpr double imPos = 0.69489971613660528327;
constexpr double sideWidth = 0.010808738133299850351;
constexpr int limitIter = 800;

int main() {
    SDL_Surface *image = createSurface(frameWidth, frameHeight);
    ColorPaletteUF palette = {};
    palette.init(0);
    const auto fastest = mandelbrotVectored<double16, long16, double, 16>;
    const auto fastestFloat = mandelbrotVectored<float8, int8, float, 8>;
    int thCount = 8;

    TIME({
             fastestFloat(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
//             antialiasImage(image, 1, 0.4);
             saveSurface(image, "fastest.png");
         }, "Fastest: ");
    const auto fastestCuda = mandelbrotRender<float>;

    TIME({
             fastestCuda(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter);
             SET_TIME;
//             antialiasImage(image, 1, 0.4);
             saveSurface(image, "fastestCuda.png");
         }, "Fastest Cuda: ");
    palette.dest();
    palette.init(1);
    SDL_FreeSurface(image);
    palette.dest();
    return 0;
}
