#include <omp.h>

#include "../../../../../../BilatTools_OMP/src/core/OMP_Tools/header/OmpTools.h"
#include "00_pi_tools.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool isPiOMPEntrelacerPromotionTab_Ok(int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static double piOMPEntrelacerPromotionTab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool isPiOMPEntrelacerPromotionTab_Ok(int n)
    {
    return isAlgoPI_OK(piOMPEntrelacerPromotionTab, n, "Pi OMP Entrelacer promotionTab");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/**
 * pattern cuda : excellent!
 */
double piOMPEntrelacerPromotionTab(int n)
    {
    const int NB_THREADS = OmpTools::setAndGetNaturalGranularity();
    double tabSumThreads[NB_THREADS];
    double somme = 0;
    double dx = 1.0 / n;

#pragma omp parallel
	{
	double sommeThreads = 0;
	const int TID = OmpTools::getTid();
	int s = TID;
	double x;
	while (s < n)
	    {
	    x = s * dx;
	    sommeThreads += fpi(x);
	    s += NB_THREADS;
	    }
	tabSumThreads[TID] = sommeThreads;
	}

    // Reduction séquentielle
    for (int i = 0; i < NB_THREADS; i++)
	{
	somme += tabSumThreads[i];
	}

    return somme * dx;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

