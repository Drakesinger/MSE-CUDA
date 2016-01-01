#include "MandelbrotJuliaMultiGPUProvider.h"
#include "MathTools.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

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

/*-----------------*\
 |*	static	   *|
 \*----------------*/

MandelbrotJuliaMultiGPU* MandelbrotJuliaMultiGPUProvider::create()
    {
    int dw = 300;
    int dh = 300;

    float dt = 2 * PI / 8000;

    int nMin = 30;
    int nMax = 100;

    return new MandelbrotJuliaMultiGPU(dw, dh, nMin, nMax);
    }

ImageFonctionel* MandelbrotJuliaMultiGPUProvider::createGL(void)
    {
    ColorRGB_01* ptrColorTitre=new ColorRGB_01(0,0,0);

    return new ImageFonctionel(create(),ptrColorTitre); // both ptr destroy by destructor of ImageFonctionel
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
