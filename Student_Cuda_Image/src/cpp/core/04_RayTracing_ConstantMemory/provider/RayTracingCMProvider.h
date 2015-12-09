#ifndef RAYTRACINGCM_PROVIDER_H_
#define RAYTRACINGCM_PROVIDER_H_

#include "cudaType.h"
#include "Image.h"
#include "Animateur.h"
#include "VariateurI.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RayTracingCMProvider
    {
    public:

	static Image* createGL(void);
	static Animable_I* createMOO(void);

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

 