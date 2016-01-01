#ifndef MANDELBROTJULIAMULTIGPU_PROVIDER_H_
#define MANDELBROTJULIAMULTIGPU_PROVIDER_H_

#include "MandelbrotJuliaMultiGPU.h"
#include "ImageFonctionel.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class MandelbrotJuliaMultiGPUProvider
    {
    public:
	static MandelbrotJuliaMultiGPU* create(void);
	static ImageFonctionel* createGL(void);
    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
