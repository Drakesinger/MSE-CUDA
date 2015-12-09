#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "Option.h"

#include "Animateur.h"
#include "AnimateurFonctionel.h"

#include "RipplingProvider.h"
#include "NewtonProvider.h"

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

int mainFreeGL(Option& option);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static void animer(Animable_I* ptrAnimable, int nbIteration);
static void animer(AnimableFonctionel_I* ptrAnimable, int nbIteration);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainFreeGL(Option& option)
    {
    cout << "\n[FreeGL] mode" << endl;

    const int NB_ITERATION = 1000;

	// Rippling
	/*{
	Animable_I* ptrRippling = RipplingProvider::create();
	animer(ptrRippling, NB_ITERATION);
	}*/

	//RayTracingSM
	/*{
	Animable_I* ptrRayTracing = RayTracingProvider::createMOO();
	animer(ptrRayTracing,NB_ITERATION);
	}*/

	// Newton
	//AnimableFonctionel_I* ptrNewton = NewtonProvider::create();
	//animer(ptrNewton, NB_ITERATION);

    cout << "\n[FreeGL] end" << endl;

    return EXIT_SUCCESS;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void animer(Animable_I* ptrAnimable, int nbIteration)
    {
    Animateur animateur(ptrAnimable, nbIteration);
    animateur.run();

    delete ptrAnimable;
    }

void animer(AnimableFonctionel_I* ptrAnimable, int nbIteration)
    {
    AnimateurFonctionel animateur(ptrAnimable, nbIteration);
    animateur.run();

    delete ptrAnimable;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

