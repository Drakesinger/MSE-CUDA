#include "Fractal.h"

/*----------------------------------------------------------------------*\
 |* Declaration *|
 \*---------------------------------------------------------------------*/

Fractal::Fractal(int n):calibreur(IntervalF(-1, 1), IntervalF(0, 1))
    {
    this->n = n;
    }

Fractal::~Fractal()
    {
    }
/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool Fractal::isDivergent(double a, double b)
    {
    return (a * a + b * b) > 4;
    }

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/


void Fractal::colorXY(uchar4* ptrColor, double x, double y, const DomaineMath& domaineMath, const int N)
    {
    int k = getK(x, y, N);

    if(k <= N)
	{
	// on scale le k sur la plage du hue [0, 1]. Similaire à un calibreur
	float hue01 = (1.0/N) * k;

	ColorTools::HSB_TO_RVB(hue01, ptrColor); // update color
	}
    else
	{
	ptrColor->x = 0;
	ptrColor->y = 0;
	ptrColor->z = 0;
	}

    ptrColor->w = 255; // opaque
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

