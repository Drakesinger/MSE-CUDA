#include <iostream>

#include "Indice2D.h"
#include "Indice1D.h"
#include "cudaTools.h"
#include "Device.h"
#include "IndiceTools.h"

#include "RipplingMath.h"

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

__global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t);

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

__global__ void rippling(uchar4* ptrDevPixels, int w, int h, float t)
    {
    RipplingMath ripplingMath = RipplingMath(w, h);

    int tid = Indice2D::tid();

    // Indice2D
    int nbThread = Indice2D::nbThread();

    // Indice1D
    //int nbThread = Indice1D::nbThread();

    const int MAX = w*h;

    int s = tid;
    /*int pixelJ = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelI = threadIdx.y + blockIdx.y * blockDim.y;
    int s = w * pixelI + pixelJ;
    ripplingMath.colorIJ(pixelI, pixelJ, t, &ptrDevPixels[s]);*/


    while(s < MAX)
	{
	int pixelI, pixelJ;
	IndiceTools::toIJ(s, w, &pixelI, &pixelJ);

	ripplingMath.colorIJ(pixelI, pixelJ, t, &ptrDevPixels[s]);

	s += nbThread;
	}
    }
/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

