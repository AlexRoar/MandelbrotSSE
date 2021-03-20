#include "src/mandutils.h"
#include "src/Graphics.h"
#include "src/ComplexSSE.h"
#include <sys/time.h>

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


const int frameWidth = 1920;
const int frameHeight = 1080;

const float rePos = -1.1959;
const float imPos = 0.3117;
const float sideWidth = 0.005;
const int limit = 256;

int main() {
    SDL_Surface * image = createSurface(frameWidth, frameHeight);
    ColorPaletteUF palette = {}; palette.init(0);
    TIME({
             mandelbrotSimple(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limit);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "set.png");
         }, "Simple");
    palette.dest();
    palette.init(1);
    TIME({
             mandelbrotSSEFl(palette, image, frameWidth, frameHeight, rePos, imPos, sideWidth, limit);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setF.png");
         }, "Float");

    palette.dest();
    palette.init(2);
    TIME({
             mandelbrotSSEFlSmooth(palette, image,frameWidth, frameHeight, rePos, imPos, sideWidth, limit);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setFSm.png");
         }, "Float smooth");

    palette.dest();
    palette.init(0);
    TIME({
             mandelbrotSSEDl(palette, image,frameWidth, frameHeight, rePos, imPos, sideWidth, limit);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setD.png");
         }, "Double");

    palette.dest();
    palette.init(0);
    TIME({
             mandelbrotSSEDlSmooth(palette, image,frameWidth, frameHeight, rePos, imPos, sideWidth, limit);
             SET_TIME;
             antialiasImage(image, 1, 0.4);
             saveSurface(image, "setDSm.png");
         }, "Double smooth");
    SDL_FreeSurface(image);
    palette.dest();
    return 0;
}
