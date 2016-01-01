#ifndef SPHERE_H_
#define SPHERE_H_

#include "cudaTools.h"
#include "Device.h"
#include "AleaTools.h"

#ifndef PI
#define PI 3.141592653589793f
#endif

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class Sphere
    {
    public:
	__host__ Sphere(float centreX, float centreY, float centreZ, float rayon, float hue)
	    {
	    // Inputs
	    this->centre.x = centreX;//centre;
	    this->centre.y = centreY;
	    this->centre.z = centreZ;
	    this->r = rayon;
	    this->hueStart = hue;
	    this->T=asin(2*hueStart-1)-((3*PI)/2);
	    // Tools
	    this->rCarre = rayon * rayon;
	    }

	/**
	 * required by example for
	 new Sphere[n]
	 */
	__host__ Sphere()
	    {
	    // rien
	    }
	__device__
	float hCarre(float2 xySol)
	    {
	    float a = (centre.x - xySol.x);
	    float b = (centre.y - xySol.y);
	    return a * a + b * b;
	    }
	__device__
	bool isEnDessous(float hCarre)
	    {
	    return hCarre < rCarre;
	    }
	__device__
	float dz(float hCarre)
	    {
	    return sqrtf(rCarre - hCarre);
	    }
	__device__
	float brightness(float dz)
	    {
	    return dz / r;
	    }
	__device__
	float distance(float dz)
	    {
	    return centre.z - dz;
	    }

	__device__
	float getHueStart()
	    {
	    return hueStart;
	    }
	__device__
	float hue(float t)
	    {
	    return 0.5 + 0.5 * sin(t +T+ 3 * PI / 2);
	    }
    private:
	// Inputs
	float r;
	float3 centre;
	float hueStart;
	float T;
	// Tools
	float rCarre;
	uchar4 padding[1];

    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
