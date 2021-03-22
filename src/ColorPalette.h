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

    constexpr static int colorsLength = 4048;
    constexpr static int pNumMax = 6;

    constexpr static ColorsPoints referencePoints5[] = {
            {0,      0xFF000000},
            {0.5,   0xFFFFFFFF},
            {0.75,      0xFF000000},
            {0.875,   0xFFFFFFFF},
            {0.9375,      0xFF000000},
            {0.96875,   0xFFFFFFFF},
            {0.984,   0xFF000000},
            {0.991,   0xFFFFFFFF},
            {0.995,   0xFF000000},
            {1,      0xFFFFFFFF},
    };

    constexpr static ColorsPoints referencePoints6[] = {
            {0,      0xFF351309},
            {0.5,   0xFF68DAFD},
            {0.75,      0xFF351309},
            {0.875,   0xFF68DAFD},
            {0.9375,      0xFF351309},
            {0.96875,   0xFF68DAFD},
            {0.984,   0xFF351309},
            {0.991,   0xFF68DAFD},
            {0.995,   0xFF351309},
            {1,      0xFF68DAFD},
    };

    constexpr static ColorsPoints referencePoints4[] = {
            {0,      0xFF640700},
            {0.16,   0xFFCB6B20},
            {0.42,   0xFFFFFFED},
            {0.6425, 0xFF00AAFF},
            {0.8575, 0xFF001241},
            {1.0, 0xFF001261},
    };

    constexpr static ColorsPoints referencePoints2[] = {
            {0,      0xFF82020e},
            {0.16,   0xFFFF1ae0},
            {0.42,   0xFFFFFaF0},
            {0.6425,   0xFFFF00F6},
            {0.8575, 0xFF06FBF8},
            {1, 0xFF0000FF},
    };

    constexpr static ColorsPoints referencePoints3[] = {
            {0,      0xFF000000},
            {0.4575,      0xFF888888},
            {1, 0xFFFFFFFF},
    };

    constexpr static ColorsPoints referencePoints1[] = {
            {0, 0xFF542517},
            {0.3, 0xFF8b3872},
            {0.5, 0xFF5872C2},
            {0.6, 0xFF69d3f5},
            {0.7, 0xFF65c6e9},
            {1,   0xFFFFFFFF},
    };


    void init(int pNo) {
        colors = static_cast<Uint32 *>(calloc(colorsLength + 1, sizeof(Uint32)));
        switch (pNo) {
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
            case 3: {
                referencePoints = referencePoints4;
                refSize = sizeof(referencePoints4) / sizeof(ColorsPoints);
                break;
            }
            case 4: {
                referencePoints = referencePoints5;
                refSize = sizeof(referencePoints5) / sizeof(ColorsPoints);
                break;
            }
            case 5: {
                referencePoints = referencePoints6;
                refSize = sizeof(referencePoints6) / sizeof(ColorsPoints);
                break;
            }
            case 0:
            default: {
                referencePoints = referencePoints1;
                refSize = sizeof(referencePoints1) / sizeof(ColorsPoints);
                break;
            }
        }
        monotonicCubicInterpolation();
    }

    void monotonicCubicInterpolation() const {
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

    static int min(int a, int b){
        return a < b ? a : b;
    }

    [[nodiscard]] Uint32 color(int i, int scale, Complex val) const {
        float nsmooth = i;
        if (val.abs() > 1)
            nsmooth = float(i + 1) - float(log2(log(val.abs())));
        int colorI = int(nsmooth / float(scale) * float(colorsLength - 1)) % colorsLength;
        return colors[colorI];
    };

    [[nodiscard]] Uint32 colorNoSmooth(int i, int scale) const {
        int colorI = int(float(i) / float(scale) * float(colorsLength - 1));
        return colors[colorI % colorsLength];
    };
};

#endif //ColorPalette_GUARD
