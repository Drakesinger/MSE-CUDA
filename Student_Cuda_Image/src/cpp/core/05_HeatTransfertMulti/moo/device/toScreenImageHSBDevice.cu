#include <iostream>

#include <stdio.h>
#include "Indice2D.h"
#include "IndiceTools.h"
#include "cudaTools.h"
#include "Device.h"

#include "ColorTools.h"

using std::cout;
using std::endl;


/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void toScreenImageHSB(uchar4* ptrDevPixels, float* ptrImageInput, int w, int h);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__global__ void toScreenImageHSB(uchar4* ptrDevPixels, float* ptrImageInput, int w, int h)
    {
    const int WH = w * h;

    const int NB_THREAD = Indice2D::nbThread();
    const int TID = Indice2D::tid();

    const float pente = (0 - 0.66) / (1.0 - -0.2);
    const float translation = 0.66 - pente * -0.2;

    int s = TID;

    while (s < WH)
	{
	uchar4 p;
	ColorTools::HSB_TO_RVB(ptrImageInput[s] * pente + translation, 1, 1, &p.x, &p.y, &p.z);
	ptrDevPixels[s] = p;
	s += NB_THREAD;
	}

    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

