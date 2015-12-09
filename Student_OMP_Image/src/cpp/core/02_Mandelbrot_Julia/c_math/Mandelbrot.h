#ifndef MANDELBROT_H_
#define MANDELBROT_H_

#include "CalibreurF.h"
#include "ColorTools.h"
#include "Fractal.h"
#include <math.h>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Dans un header only pour preparer la version cuda
 */
class Mandelbrot : public Fractal
    {
	/*--------------------------------------*\
	 |*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	/**
	 * calibreurColor : transformation affine entre [-1,1] (l'output de f(x,y)) et [0,1] (le spectre hsb)
	 */
	Mandelbrot(int n) : Fractal(n)
	    {
		this->n = n;
	    }

	virtual ~Mandelbrot(void)
	    {
	    // nothing
	    }

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    protected:

	virtual int getK(double x, double y, int N)
	    {
		float a = 0;
		float b = 0;

		int k = 0;

		while(!isDivergent(a, b) && k <= N)
		    {
		    float aCopy = a;
		    a = (aCopy*aCopy - b*b) + x;
		    b = 2.0 * aCopy * b + y;

		    k++;
		    }

		return k;
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:
	int n;
    }
;

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
