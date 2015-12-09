#ifndef JULIA_H_
#define JULIA_H_

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
class Julia : public Fractal
    {
	/*--------------------------------------*\
	 |*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	/**
	 * calibreurColor : transformation affine entre [-1,1] (l'output de f(x,y)) et [0,1] (le spectre hsb)
	 */
	Julia(float c1, float c2) : Fractal(n)
	    {
	    this->c1 = c1;
	    this->c2 = c2;
	    }

	virtual ~Julia(void)
	    {
	    // nothing
	    }

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    protected:

	virtual int getK(double x, double y, int N)
		    {
		    float a = x;
		    float b = y;

		    int k = 0;

		    while(!isDivergent(a, b) && k <= N)
			{
			float aCopy = a;
			a = (aCopy*aCopy - b*b) + c1;
			b = 2.0 * aCopy * b + c2;

			k++;
			}

		    return k;
		    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	double c1;
	double c2;
    }
;

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
