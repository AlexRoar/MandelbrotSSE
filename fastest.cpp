#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include <sys/time.h>


constexpr int frameWidth = 1920 * 16;
constexpr int frameHeight = 1080 * 16;

constexpr double rePos = -0.027844723880907473745;
constexpr double imPos = 0.69489971613660528327;
constexpr double sideWidth = 0.010808738133299850351;
constexpr int limitIter = 800;

int main() {
    SDL_Surface *image = createSurface(frameWidth, frameHeight);
    ColorPaletteUF palette = {};
    palette.init(0);
    const auto fastest = mandelbrotVectored<double16, long16, double, 16>;

    int thCount = 8;

    TIME({
             fastest(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
//             antialiasImage(image, 1, 0.4);
             saveSurface(image, "fastest.png");
         }, "\nFastest: ");
    palette.dest();
    palette.init(1);
    SDL_FreeSurface(image);
    palette.dest();
    return 0;
}
