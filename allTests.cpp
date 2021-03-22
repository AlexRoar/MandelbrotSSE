#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include <sys/time.h>


constexpr int frameWidth = 1920;
constexpr int frameHeight = 1080;

constexpr double rePos = -0.021443584518319098;
constexpr double imPos = 0.7103940042262454;
constexpr double sideWidth = 0.0000317024843773815;
constexpr int limitIter = 256;

int main() {
    SDL_Surface *plt = showPalette(0);
    saveSurface(plt, "plt0.png");
    SDL_FreeSurface(plt);

    plt = showPalette(1);
    saveSurface(plt, "plt1.png");
    SDL_FreeSurface(plt);
    plt = showPalette(2);
    saveSurface(plt, "plt2.png");
    SDL_FreeSurface(plt);

    plt = showPalette(3);
    saveSurface(plt, "plt3.png");
    SDL_FreeSurface(plt);

    plt = showPalette(4);
    saveSurface(plt, "plt4.png");
    SDL_FreeSurface(plt);

    plt = showPalette(5);
    saveSurface(plt, "plt5.png");
    SDL_FreeSurface(plt);

    SDL_Surface *image = createSurface(frameWidth, frameHeight);
    ColorPaletteUF palette = {};
    palette.init(0);

    const auto vectoredDouble = mandelbrotVectored<double2, long2, double, 2>;
    const auto vectoredDouble4 = mandelbrotVectored<double4, long4, double, 4>;
    const auto vectoredDouble8 = mandelbrotVectored<double8, long8, double, 8>;
    const auto vectoredDouble16 = mandelbrotVectored<double16, long16, double, 16>;
    const auto vectoredDouble32 = mandelbrotVectored<double32, long32, double, 32>;
    const auto vectoredFloat2 = mandelbrotVectored<float2, int2, float, 2>;
    const auto vectoredFloat4 = mandelbrotVectored<float4, int4, float, 4>;
    const auto vectoredFloat8 = mandelbrotVectored<float8, int8, float, 8>;
    const auto vectoredFloat16 = mandelbrotVectored<float16, int16, float, 16>;
    const auto vectoredFloat32 = mandelbrotVectored<float32, int32, float, 32>;
    const auto vectoredFloat512 = mandelbrotVectored<float512, int512, float, 512>;

    int thCount = 1;

    TIME({
             mandelbrotSimple(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "set.png");
         }, "Simple");
    TIME({
             vectoredFloat2(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVF2.png");
         }, "Simple vector float2");
    TIME({
             vectoredFloat4(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVF4.png");
         }, "Simple vector float4");
    TIME({
             vectoredFloat8(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVF8.png");
         }, "Simple vector float8");
    TIME({
             vectoredFloat16(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVF16.png");
         }, "Simple vector float16");
    TIME({
             vectoredFloat32(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVF32.png");
         }, "Simple vector float32");
    TIME({
             vectoredFloat512(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVF512.png");
         }, "Simple vector float512");
    TIME({
             vectoredDouble(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVD2.png");
         }, "Simple vector double2");
    TIME({
             vectoredDouble4(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVD4.png");
         }, "Simple vector double4");
    TIME({
             vectoredDouble8(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVD8.png");
         }, "Simple vector double8");
    TIME({
             vectoredDouble16(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVD16.png");
         }, "Simple vector double16");
    TIME({
             vectoredDouble32(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter, thCount);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setVD32.png");
         }, "Simple vector double32");
    palette.dest();
    palette.init(1);
    TIME({
             mandelbrotSSEFl(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setF.png");
         }, "Float SSE");

    palette.dest();
    palette.init(1);
    TIME({
             mandelbrotSSEFlSmooth(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setFSm.png");
         }, "Float SSE smooth");

    palette.dest();
    palette.init(2);
    TIME({
             mandelbrotSSEDl(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setD.png");
         }, "Double SSE");

    palette.dest();
    palette.init(2);
    TIME({
             mandelbrotSSEDlSmooth(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limitIter);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setDSm.png");
         }, "Double SSE smooth");
    SDL_FreeSurface(image);
    palette.dest();
    return 0;
}
