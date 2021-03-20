//
// Created by Александр Дремов on 19.03.2021.
//

#ifndef ColorPalette_GUARD
#define ColorPalette_GUARD

#include <cstdlib>
#include <SDL.h>
#include "Complex.h"

struct ColorPaletteUF {
    struct ColorsPoints {
        float pos;
        Uint32 color;
    };

    Uint32 *colors;
    const ColorsPoints* referencePoints;
    size_t refSize;

    constexpr static int colorsLength = 2048;
    constexpr static int pNumMax = 3;
    constexpr static ColorsPoints referencePoints1[] = {
            {0,      0xFF640700},
            {0.16,   0xFFCB6B20},
            {0.42,   0xFFFFFFED},
            {0.6425, 0xFF00AAFF},
            {0.8575, 0xF0001241},
            {1.0, 0xF0001261},
    };

    constexpr static ColorsPoints referencePoints2[] = {
            {0,      0xFF82020e},
            {0.16,   0xFFFF1ae0},
            {0.42,   0xFFFFFaF0},
            {0.6425,   0xFFFF00F6},
            {0.8575, 0xFF06FBF8},
            {1.0, 0xFF0000FF},
    };

    constexpr static ColorsPoints referencePoints3[] = {
            {0,      0xFFFFFFFF},
            {0.8575,      0xFF888888},
            {1.0, 0xFF000000},
    };

    void init(int pNo) {
        colors = static_cast<Uint32 *>(calloc(colorsLength + 1, sizeof(Uint32)));
        switch (pNo) {
            case 0: {
                referencePoints = referencePoints1;
                refSize = sizeof(referencePoints1) / sizeof(ColorsPoints);
                break;
            }
            case 1: {
                referencePoints = referencePoints2;
                refSize = sizeof(referencePoints2) / sizeof(ColorsPoints);
                break;
            }
            case 2: {
                referencePoints = referencePoints3;
                refSize = sizeof(referencePoints3) / sizeof(ColorsPoints);
                break;
            }
            default: {
                referencePoints = referencePoints1;
                refSize = sizeof(referencePoints1) / sizeof(ColorsPoints);
                break;
            }
        }
        monotonicCubicInterpolation();

    }

    void monotonicCubicInterpolation() {
        struct CalcData {
            double dxs, dys, ms;
        };
        struct PolyCoef {
            double cs1, cs2, cs3;
        };

        auto *calcData = static_cast<CalcData *>(calloc(refSize + 1, sizeof(CalcData)));
        auto *coef = static_cast<PolyCoef *>(calloc(refSize + 3, sizeof(PolyCoef)));

        Uint32 initialMask = 0x000000FF;
        for (int m_shift = 0; m_shift < sizeof(Uint32); m_shift++) {
            for (int i = 0; i < refSize - 1; i++) {
                double dx = referencePoints[i + 1].pos - referencePoints[i].pos;
                double dy = double(referencePoints[i + 1].color & initialMask) -
                            double(referencePoints[i].color & initialMask);
                calcData[i] = {dx, dy, dy / dx};
            }

            coef[0].cs1 = calcData[0].ms;
            for (int i = 0; i < refSize - 1; i++) {
                double m = calcData[i].ms, mNext = calcData[i + 1].ms;
                if (m * mNext <= 0) {
                    coef[i + 1].cs1 = 0;
                } else {
                    double dx_ = calcData[i].dxs;
                    double dxNext = calcData[i + 1].dxs;
                    double common = dx_ + dxNext;
                    coef[i + 1].cs1 = 3 * common / ((common + dxNext) / m + (common + dx_) / mNext);
                }
            }
            coef[refSize - 1].cs1 = calcData[refSize - 1].ms;

            for (int i = 0; i < refSize; i++) {
                double c1 = coef[i].cs1, m_ = calcData[i].ms, invDx = 1 / calcData[i].dxs;
                double common_ = c1 + coef[i + 1].cs1 - m_ - m_;
                coef[i].cs2 = (m_ - c1 - common_) * invDx;
                coef[i].cs3 = common_ * invDx * invDx;
            }

            for (int i = 0; i < colorsLength; i++) {
                double pos = double(i) / double(colorsLength - 1);
                if (i == colorsLength - 1){
                    colors[i] = referencePoints[refSize - 1].color;
                    continue;
                }
                int point = 0;
                for (; point < refSize; point++) {
                    if (pos >= referencePoints[point].pos && pos < referencePoints[point + 1].pos) {
                        break;
                    }
                }

                double diff = pos - referencePoints[point].pos;
                double diffSq = diff * diff;
                double interpolated = (referencePoints[point].color & initialMask) + coef[point].cs1 * diff +
                                      coef[point].cs2 * diffSq + coef[point].cs3 * diff * diffSq;
                colors[i] |= Uint32(interpolated) & initialMask;
            }

            initialMask <<= sizeof(char) * 8;
        }
        free(coef);
        free(calcData);
    }

    void dest() {
        free(colors);
    }

    static ColorPaletteUF *New() {
        auto *thou = static_cast<ColorPaletteUF *>(calloc(1, sizeof(ColorPaletteUF)));
        thou->init(0);
        return thou;
    }

    void Delete() {
        dest();
        free(this);
    }

    static double nonLinearity(double x){
        return log(x + 1);
    }

    [[nodiscard]] Uint32 color(int i, int scale, Complex val) const {
        double smoothed = log2(log2(val.absNoSqrt()) / 2);
        int colorI = (int)(nonLinearity((double(i) + 15.0 - smoothed) / scale) / nonLinearity(1) *  colorsLength) % (colorsLength);
        return colors[colorI];
    };

    [[nodiscard]] Uint32 colorNoSmooth(int i, int scale) const {
        int colorI = (int)(sqrt(i + 10 + 0.4) * scale) % (colorsLength);
        return colors[colorI];
    };
};

#endif //ColorPalette_GUARD
