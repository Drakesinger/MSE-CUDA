#ifndef RAYTRACING_MATH_H_
#define RAYTRACING_MATH_H_

#include "CalibreurF.h"
#include "ColorTools.h"
#include "Sphere.h"
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
class RayTracingMath
    {
	/*--------------------------------------*\
	 |*		Constructeur		*|
	 \*-------------------------------------*/

    public:
	__device__ RayTracingMath(int _n):
	calibreur(IntervalF(0.0, PI/2), IntervalF(0, 1))
	    {
	    this->n = _n;
	    }
	__device__
	    virtual ~RayTracingMath(void)
	    {
	    // rien
	    }

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:
	__device__
	void colorXY(uchar4* ptrColor, Sphere* ptrDevSphere, int nbSphere, float x, float y,float t)
	    {

	    float2 xy;
	    xy.x = x;
	    xy.y = y;
	    float h = 0;
	    float b = 0;
	    float smallestDistance = 999999;
	    for (int i = 0; i < nbSphere; i++)
		{

		float h2 = ptrDevSphere[i].hCarre(xy);
		if (ptrDevSphere[i].isEnDessous(h2))
		    {
		    float dZ = ptrDevSphere[i].dz(h2);
		    float dist = ptrDevSphere[i].distance(dZ);
		    if (dist < smallestDistance)
			{
			smallestDistance = dist;
			h = ptrDevSphere[i].hue(t);
			b = ptrDevSphere[i].brightness(dZ);
			}
		    }

		}

	    calibreur.calibrer(h);
	    ColorTools::HSB_TO_RVB(h,1,b,ptrColor);

	    ptrColor->w = 255; // opaque

	    }

    protected:
	__device__
	float h(int k)
	    {
	    return (1.0 / this->n) * k;
	    }

	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

	// Inputs
	int n;

	//tools
	CalibreurF calibreur;

    }
;

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
