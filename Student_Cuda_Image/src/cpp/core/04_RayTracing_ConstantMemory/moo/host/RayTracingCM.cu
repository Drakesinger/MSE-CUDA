#include <iostream>
#include <assert.h>

#include "IntervalF_GPU.h"
#include "RayTracingCM.h"
#include "Device.h"
#include "cudaTools.h"
#include "AleaTools.h"
#include "ConstantMemoryLink.h"


using std::cout;
using std::endl;



/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

#define NBSPHERE 50
//ou const int LENGTH =2;
__constant__ Sphere tabSphere_CM[NBSPHERE];




/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void raytracingCM(uchar4* ptrDevPixels, Sphere* ptrDevSphere, int nbSphere, int w, int h, float t);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/


ConstantMemoryLink constantMemoryLink(void)
{
    Sphere* ptrDevTabData;
    size_t sizeAll = NBSPHERE * sizeof(Sphere);
    HANDLE_ERROR(cudaGetSymbolAddress((void **)
    &ptrDevTabData, tabSphere_CM));
    ConstantMemoryLink cmLink =
    {
	    (void**) ptrDevTabData, NBSPHERE, sizeAll
    };
    return cmLink;
}

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

RayTracingCM::RayTracingCM(int w, int h)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->t=0;

    //Tools
    this->dg = dim3(32, 4, 1);
    this->db = dim3(64, 8, 1);
    this->tabValue = new Sphere[NBSPHERE];

    AleaTools aleaTools = AleaTools();
    float bord = 200;
    for(int i=0;i<NBSPHERE;i++)
	{
	float rayon = aleaTools.uniformeAB(20, this->w/10);
	float x = aleaTools.uniformeAB(bord, this->h -bord);
	float y = aleaTools.uniformeAB(bord, this->w -bord);
	float z = aleaTools.uniformeAB(10, 2* this->w);
	float hue = aleaTools.uniforme01();
	tabValue[i] = *(new Sphere(x,y,z,rayon,hue));
	}

    // Outputs
    this->title = "RayTracing Constant Memory";

    //print(dg, db);
    Device::assertDim(dg, db);

    // Recupération de la constantMemory CM du device
    ConstantMemoryLink cmLink = constantMemoryLink();
    ptrDevTabData = (Sphere*)cmLink.ptrDevTab;
    size_t sizeALL = cmLink.sizeAll;
    // transfert Host->Device
    HANDLE_ERROR(cudaMemcpy(ptrDevTabData,
	    tabValue,sizeALL,cudaMemcpyHostToDevice));
    }

RayTracingCM::~RayTracingCM()
    {
    delete[] tabSphere_CM;
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

/**
 * Override
 */
void RayTracingCM::process(uchar4* ptrDevPixels, int w, int h)
{
    raytracingCM<<<dg,db>>>(ptrDevPixels,ptrDevTabData,NBSPHERE, w, h, this->t);
}

/**
 * Override
 */
void RayTracingCM::animationStep()
{
    t+=0.1;
}

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
float RayTracingCM::getAnimationPara(void)
{
//return n;
//variateurN.get();
    return t;
}

/**
 * Override
 */
int RayTracingCM::getW(void)
{
return w;
}

/**
 * Override
 */
int RayTracingCM::getH(void)
{
return h;
}

/**
 * Override
 */
string RayTracingCM::getTitle(void)
{
return title;
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/



/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

