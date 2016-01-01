#ifndef RAYTRACINGSM_PROVIDER_H_
#define RAYTRACINGSM_PROVIDER_H_

#include "cudaType.h"
#include "Image.h"
#include "VariateurI.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RayTracingSMProvider
    {
    public:

	static Image* createGL(void);
	static Animable_I* createMOO(void);

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

