#ifndef SRC_CPP_CORE_02_MANDELBROT_JULIA_JOE_C_MATH_FRACTAL_H_
#define SRC_CPP_CORE_02_MANDELBROT_JULIA_JOE_C_MATH_FRACTAL_H_

#include "CalibreurF.h"
#include "ColorTools.h"
#include <math.h>
#include "Fractal.h"
#include "DomaineMath.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class Fractal
    {
    public:
	Fractal(int n);
	virtual ~Fractal();
	void colorXY(uchar4* ptrColor, double x, double y, const DomaineMath& domaineMath, const int N);

    protected:
	int n;
	bool isDivergent(double a, double b);
	virtual int getK(double x, double y, int N) = 0;
	// Tools
	CalibreurF calibreur;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
