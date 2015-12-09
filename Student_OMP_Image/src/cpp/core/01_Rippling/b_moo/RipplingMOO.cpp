#include <iostream>
#include <omp.h>

#include "RipplingMOO.h"
#include "OmpTools.h"
#include "IndiceTools.h"
#include "RipplingMath.h"

using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

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

RipplingMOO::RipplingMOO(unsigned int w, unsigned int h, float dt)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->dt = dt;

    // Tools
    this->t = 0;
    this->parallelPatern=OMP_MIXTE;
    }

RipplingMOO::~RipplingMOO(void)
    {
    // rien
    }

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Override
 */
void RipplingMOO::process(uchar4* ptrTabPixels, int w, int h)
    {
    switch (parallelPatern)
	{

	case OMP_ENTRELACEMENT: // Plus lent sur CPU
	    {
	    entrelacementOMP(ptrTabPixels, w, h);
	    break;
	    }

	case OMP_FORAUTO: // Plus rapide sur CPU
	    {
	    forAutoOMP(ptrTabPixels, w, h);
	    break;
	    }

	case OMP_MIXTE: // Pour tester que les deux implementations fonctionnent
	    {
	    // Note : Des saccades peuvent apparaitre � cause de la grande difference de fps entre la version entrelacer et auto
	    static bool isEntrelacement = true;
	    if (isEntrelacement)
		{
		entrelacementOMP(ptrTabPixels, w, h);
		}
	    else
		{
		forAutoOMP(ptrTabPixels, w, h);
		}
	    isEntrelacement = !isEntrelacement; // Pour swithcer a chaque iteration
	    break;
	    }
	}
    }

/**
 * Override
 */
void RipplingMOO::animationStep()
    {
    t += dt;
    }

/*--------------*\
 |*	get	*|
 \*-------------*/

/**
 * Override
 */
float RipplingMOO::getAnimationPara()
    {
    return t;
    }

/**
 * Override
 */
int RipplingMOO::getW()
    {
    return w;
    }

/**
 * Override
 */
int RipplingMOO::getH()
    {
    return h;
    }

/**
 * Override
 */
string RipplingMOO::getTitle()
    {
    return "Rippling_OMP";
    }

/*-------------*\
 |*     set	*|
 \*------------*/

void RipplingMOO::setParallelPatern(ParallelPatern parallelPatern)
    {
    this->parallelPatern=parallelPatern;
    }


/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/**
 * Code entrainement Cuda
 */
void RipplingMOO::entrelacementOMP(uchar4* ptrTabPixel, int w, int h)
    {

    RipplingMath ripplingMath(w,h); // ici pour preparer cuda

        const int WH = w * h;

        #pragma omp parallel
        	{
        	const int NB_THREAD = OmpTools::getNbThread(); // dans region parallel

        	const int TID = OmpTools::getTid();
        	int s = TID; // in [0,...

        	int i;
        	int j;
        	while (s < WH)
        	    {
        	    IndiceTools::toIJ(s, w, &i, &j); // s[0,W*H[ --> i[0,H[ j[0,W[

        	    ripplingMath.colorIJ(&ptrTabPixel[s], i, j,t);

        	    s += NB_THREAD;
        	    }
        	}

    }

/**
 * Code naturel et direct OMP
 */
void RipplingMOO::forAutoOMP(uchar4* ptrTabPixels, int w, int h)
    {

    RipplingMath ripplingMath(w, h);

       #pragma omp parallel for
           for (int i = 0; i < h; i++)
       	{
       	for (int j = 0; j < w; j++)
       	    {
       	    // int s = i * W + j;
       	    int s = IndiceTools::toS(w, i, j);    // i[0,H[ j[0,W[  --> s[0,W*H[

       	    ripplingMath.colorIJ(&ptrTabPixels[s], i, j, t);
       	    }
       	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
