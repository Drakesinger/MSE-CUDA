#include <iostream>
#include <assert.h>

#include "IntervalF_GPU.h"
#include "RayTracingSM.h"
#include "Device.h"
#include "cudaTools.h"
#include "AleaTools.h"
#include "ConstantMemoryLink.h"


using std::cout;
using std::endl;



/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/




/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void raytracingSM(uchar4* ptrDevPixels, Sphere* ptrDevSphere, int nbSphere, int w, int h, float t);

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

RayTracingSM::RayTracingSM(int w, int h)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->t=0;
    nbSphere=50;
    //Tools
    this->dg = dim3(32, 4, 1);
    this->db = dim3(64, 8, 1);
    this->tabValue = new Sphere[nbSphere];
    AleaTools aleaTools = AleaTools();
    float bord = 200;
    for(int i=0;i<nbSphere;i++)
	{
	float rayon = aleaTools.uniformeAB(20, this->w/10);
	float x = aleaTools.uniformeAB(bord, this->h -bord);
	float y = aleaTools.uniformeAB(bord, this->w -bord);
	float z = aleaTools.uniformeAB(10, 2* this->w);
	float hue = aleaTools.uniforme01();
	tabValue[i] = Sphere(x,y,z,rayon,hue);
	}

    // Outputs
    this->title = "RayTracing Shared Memory";

    //print(dg, db);
    Device::assertDim(dg, db);

    }

RayTracingSM::~RayTracingSM()
    {
    delete[] tabValue;
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

/**
 * Override
 */
void RayTracingSM::process(uchar4* ptrDevPixels, int w, int h)
{


    Sphere* ptrDevSphere=NULL;
    size_t size = nbSphere*sizeof(Sphere);
    HANDLE_ERROR(cudaMalloc(&ptrDevSphere,size));
    HANDLE_ERROR(cudaMemcpy(ptrDevSphere, tabValue,size,cudaMemcpyHostToDevice));
    raytracingSM<<<dg,db,size>>>(ptrDevPixels,ptrDevSphere,this->nbSphere, w, h, this->t);

    HANDLE_ERROR(cudaFree(ptrDevSphere));
}

/**
 * Override
 */
void RayTracingSM::animationStep()
{
    t+=0.1;
}

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
float RayTracingSM::getAnimationPara(void)
{
    return t;
}

/**
 * Override
 */
int RayTracingSM::getW(void)
{
return w;
}

/**
 * Override
 */
int RayTracingSM::getH(void)
{
return h;
}

/**
 * Override
 */
string RayTracingSM::getTitle(void)
{
return title;
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/



/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

