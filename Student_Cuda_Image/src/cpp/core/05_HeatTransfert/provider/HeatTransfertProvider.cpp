#include "HeatTransfertProvider.h"
#include "MathTools.h"
#include "DomaineMath.h"
#include <limits>

HeatTransfert* HeatTransfertProvider::create()
    {
    int dw = 300;
    int dh = 300;

    int blindIteration = 10;
    float k = 0.25f;

    return new HeatTransfert(dw, dh, blindIteration, k);
    }

Image* HeatTransfertProvider::createGL(void)
    {
    ColorRGB_01* ptrColorTitre = new ColorRGB_01(0, 0, 0);
    return new Image(create(),ptrColorTitre);
    }
