#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "GLUTImageViewers.h"
#include "Option.h"
#include "Viewer.h"
#include "ViewerZoomable.h"

#include "RipplingProvider.h"
#include "MandelbrotProvider.h"
#include "MandelbrotJuliaMultiGPUProvider.h"
#include "NewtonProvider.h"
#include "RayTracingProvider.h"
#include "RayTracingSMProvider.h"
#include "RayTracingCMProvider.h"
#include "HeatTransfertProvider.h"
#include "HeatTransfertProviderM.h"


using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainGL(Option& option);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainGL(Option& option)
    {
    cout << "\n[OpenGL] mode" << endl;

    GLUTImageViewers::init(option.getArgc(), option.getArgv());
    Image* ptrRippling = RipplingProvider::createGL();
    ImageFonctionel* ptrMandel = MandelbrotProvider::createGL();
    ImageFonctionel* ptrNewton = NewtonProvider::createGL();
    Image* ptrRayTracing = RayTracingProvider::createGL();
    Image* ptrRayTracingSM = RayTracingSMProvider::createGL();
    Image* ptrRayTracingCM = RayTracingCMProvider::createGL();
    //Image* ptrHeatTransfert = HeatTransfertProvider::createGL();
    ImageFonctionel* ptrMandelMultiGPU = MandelbrotJuliaMultiGPUProvider::createGL();
    Image* ptrHeatTransfertMulti = HeatTransfertProviderM::createGL();


    GLUTImageViewers ripplingViewer(ptrRippling, true, false, 0, 0);
    GLUTImageViewers mandelbrotViewer(ptrMandel, true, false, 300, 0);
    GLUTImageViewers newtonViewer(ptrNewton,true, true, 600, 0);
    GLUTImageViewers rayTracingViewer(ptrRayTracing,true,false,900,0);
    GLUTImageViewers rayTracingSMViewer(ptrRayTracingSM,true,false,0,300);
    GLUTImageViewers rayTracingCMViewer(ptrRayTracingCM,true,false,300,300);
    //GLUTImageViewers ptrHeatTransfertViewer(ptrHeatTransfert,true,false,300,600);
    GLUTImageViewers mandelbrotMultiViewer(ptrMandelMultiGPU, true, false, 600, 300);
    GLUTImageViewers ptrHeatTransfertViewerMulti(ptrHeatTransfertMulti,true,false,900,300);

    GLUTImageViewers::runALL(); // Bloquant, Tant qu'une fenetre est ouverte


    {
    	delete ptrRippling;
    	delete ptrMandel;
    	delete ptrNewton;
    	delete ptrRayTracing;
    	delete ptrRayTracingSM;
    	delete ptrRayTracingCM;
    	//delete ptrHeatTransfert;
    	delete ptrHeatTransfertMulti;
    	delete ptrMandelMultiGPU;

    	ptrRippling = NULL;
    	ptrMandel = NULL;
    	ptrMandelMultiGPU = NULL;
    	ptrNewton = NULL;
    	ptrRayTracing = NULL;
    	ptrRayTracingSM = NULL;
    	ptrRayTracingCM = NULL;
    	//ptrHeatTransfert = NULL;
    	ptrHeatTransfertMulti = NULL;
    }

    return EXIT_SUCCESS;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/


