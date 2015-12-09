#include <iostream>
#include <assert.h>

#include "Rippling.h"
#include "Device.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t);

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

Rippling::Rippling(int w, int h, float dt)
    {

    // Inputs
    this->w = w;
    this->h = h;
    this->dt = dt;

    // Tools
    this->dg = dim3(32,32,1);
    this->db = dim3(64,16,1);

    // Indice1D
    //this->dg = dim3(256,1,1);
    //this->db = dim3(256,1,1);

    // 1 - 1
    //this->dg = dim3(512,512,1);
    //this->db = dim3(2,2,1);

    this->t = 0;

    // Outputs
    this->title = "Rippling_Cuda";

    //print(dg, db);
    //Device::assertDim(dg, db);
    }

Rippling::~Rippling()
    {
    // rien
    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/


/**
 * Override
 */
void Rippling::process(uchar4* ptrDevPixels, int w, int h)
    {
    // TODO lancer le kernel avec <<<dg,db>>>
    rippling<<<dg, db>>>(ptrDevPixels, w, h, t);
    }


/**
 * Override
 */
void Rippling::animationStep()
    {
    // TODO
    this->t += this->dt;
    }

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
float Rippling::getAnimationPara(void)
    {
    return t;
    }

/**
 * Override
 */
int Rippling::getW(void)
    {
    return w;
    }

/**
 * Override
 */
int Rippling::getH(void)
    {
    return  h;
    }

/**
 * Override
 */
string Rippling::getTitle(void)
    {
    return title;
    }


/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

