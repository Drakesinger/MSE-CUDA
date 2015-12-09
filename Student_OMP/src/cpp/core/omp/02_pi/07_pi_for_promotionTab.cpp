#include <omp.h>
#include "00_pi_tools.h"
#include "MathTools.h"
#include "OmpTools.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool isPiOMPforPromotionTab_Ok(int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static double piOMPforPromotionTab(int n);
static void syntaxeSimplifier(double* tabSumThread, int n);
static void syntaxeFull(double* tabSumThread, int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

bool isPiOMPforPromotionTab_Ok(int n)
    {
    return isAlgoPI_OK(piOMPforPromotionTab, n, "Pi OMP for promotion tab");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/**
 * De-synchronisation avec PromotionTab
 */
double piOMPforPromotionTab(int n)
    {
    double somme = 0;
    double dx = 1.0 / n;
    double x;
    const int NB_THREAD = OmpTools::setAndGetNaturalGranularity();
    double tabPromo[NB_THREAD];

    for (int i = 0; i < NB_THREAD; i++)
	tabPromo[i] = 0;


#pragma omp parallel for private (x)
    for (int i = 0; i < n; i++)
	{
	x = i * dx;
	const int TID = OmpTools::getTid();
	tabPromo[TID] += fpi(x);
	}

    for (int i = 0; i < NB_THREAD; i++)
	{
	somme += tabPromo[i];
	}

    return somme / n;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

