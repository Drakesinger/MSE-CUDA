#include <iostream>
#include <assert.h>

#include "IntervalF_GPU.h"
#include "RayTracing.h"
#include "Device.h"
#include "cudaTools.h"
#include "AleaTools.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void raytracing(uchar4* ptrDevPixels, Sphere* ptrDevSphere, int nbSphere, int w, int h, float t);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

RayTracing::RayTracing(int w, int h)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->t=0;

    //Tools
    this->dg = dim3(16, 2, 1);
    this->db = dim3(32, 4, 1);
    this->nbSphere = 50;
    this->tabSphere = new Sphere[nbSphere];

    AleaTools aleaTools = AleaTools();
    float bord = 200;
    for(int i=0;i<nbSphere;i++)
	{
	//TODO random
	float rayon = aleaTools.uniformeAB(20, this->w/10);
	float x = aleaTools.uniformeAB(bord, this->h -bord);
	float y = aleaTools.uniformeAB(bord, this->w -bord);
	float z = aleaTools.uniformeAB(10, 2* this->w);
	float hue = aleaTools.uniforme01();
	tabSphere[i] = Sphere(x,y,z,rayon,hue);
	}

    // Outputs
    this->title = "RayTracing_Cuda";

    //print(dg, db);
    Device::assertDim(dg, db);
    }

RayTracing::~RayTracing()
    {
    delete[] tabSphere;
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

/**
 * Override
 */
void RayTracing::process(uchar4* ptrDevPixels, int w, int h)
{
    Sphere* ptrDevSphere=NULL;
    size_t size = nbSphere*sizeof(Sphere);
    HANDLE_ERROR(cudaMalloc(&ptrDevSphere,size));
    HANDLE_ERROR(cudaMemcpy(ptrDevSphere, this->tabSphere,size,cudaMemcpyHostToDevice));
    raytracing<<<dg,db>>>(ptrDevPixels,ptrDevSphere,this->nbSphere, w, h, this->t);

    HANDLE_ERROR(cudaFree(ptrDevSphere));
}

/**
 * Override
 */
void RayTracing::animationStep()
{
    t+=3.14/200;
}

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
float RayTracing::getAnimationPara(void)
{
//return n;
//variateurN.get();
    return t;
}

/**
 * Override
 */
int RayTracing::getW(void)
{
return w;
}

/**
 * Override
 */
int RayTracing::getH(void)
{
return h;
}

/**
 * Override
 */
string RayTracing::getTitle(void)
{
return title;
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

