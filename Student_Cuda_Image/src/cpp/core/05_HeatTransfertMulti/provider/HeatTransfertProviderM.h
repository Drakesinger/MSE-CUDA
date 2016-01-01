#ifndef HEATTRANSFERT_PROVIDERM_H_
#define HEATTRANSFERT_PROVIDERM_H_

#include "HeatTransfertM.h"
#include "Image.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class HeatTransfertProviderM
    {
    public:

	static HeatTransfertM* createMOO(void);
	static Image* createGL(void);

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

