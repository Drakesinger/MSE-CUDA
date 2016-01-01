#include <iostream>

#include "Indice2D.h"
#include "IndiceTools.h"
#include "cudaTools.h"
#include "Device.h"

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

__global__ void ecrasement(float* ptrImageInOutput, float* ptrImageHeater, float* ptrImageOutput, int w, int h);

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

__global__ void ecrasement(float* ptrImageInOutput, float* ptrImageHeater, float* ptrImageOutput, int w, int h)
    {
    const int WH = w * h;

    const int NB_THREAD = Indice2D::nbThread();
    const int TID = Indice2D::tid();

    int s = TID;

    while (s < WH)
	{

	bool t = ptrImageHeater[s] != 0;

	ptrImageOutput[s] = t * ptrImageHeater[s] + (1-t) * ptrImageInOutput[s];

	s += NB_THREAD;
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

