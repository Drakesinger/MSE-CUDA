#include <iostream>
#include <assert.h>

#include "DomaineMath.h"
#include "HeatTransfert.h"
#include "Device.h"
#include <math.h>
#include <algorithm>
#include "MathTools.h"

using std::cout;
using std::endl;

extern __global__ void heatDiffusion(float* ptrDevIn, float* ptrDevOut, float k, int w, int h);
extern __global__ void heatEcrasement(float* ptrDevCurrent, float* ptrDevH, int w, int h);
extern __global__ void heatImageHSB(float* ptrDevIn, uchar4* ptrDevOut, CalibreurF calibreur, int w, int h);

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

HeatTransfert::HeatTransfert(int w, int h, int blindIterations, float k) :
	calibreur(IntervalF(0, 1), IntervalF(0.7, 0)), isAInput(true), nbIteration(blindIterations), nbBlindIteration(blindIterations), k(k), ptrH(
		new float[w * h]), ptrA(new float[w * h]), ptrB(new float[w * h])
    {
    std::fill(ptrH, ptrH + w * h, 0);
    std::fill(ptrB, ptrB + w * h, 0);

    // Inputs
    this->w = w;
    this->h = h;
    this->t =0;

    this->dg = dim3(256, 1, 1);
    this->db = dim3(256, 1, 1);

    // Outputs
    this->title = "HeatTransfert Cuda";

    // fill/initialize the image
    for (int i = 300; i <= 500; ++i)
	for (int j = 300; j <= 500; ++j)
	    ptrH[i * w + j] = 1;

    for (int j = 179; j <= 195; ++j)
	{
	for (int i = 179; i <= 195; ++i)
	    ptrH[i * w + j] = 0.2;
	for (int i = 605; i <= 621; ++i)
	    ptrH[i * w + j] = 0.2;
	}
    for (int j = 605; j <= 621; ++j)
	{
	for (int i = 179; i <= 195; ++i)
	    ptrH[i * w + j] = 0.2;
	for (int i = 605; i <= 621; ++i)
	    ptrH[i * w + j] = 0.2;
	}

    ptrH[295 * w + 400] = ptrH[505 * w + 400] = ptrH[400 * w + 505] = ptrH[400 * w + 295] = 0.2f;

    for (int i = 0; i < w * h; ++i)
	ptrA[i] = ptrH[i];

    instanciateHeatDevImage(&ptrH, &ptrDevH, w * h * sizeof(float));
    instanciateHeatDevImage(&ptrA, &ptrDevA, w * h * sizeof(float));
    instanciateHeatDevImage(&ptrB, &ptrDevB, w * h * sizeof(float));

    }

HeatTransfert::~HeatTransfert()
    {
    destructHeatDevImage(&ptrDevA);
    destructHeatDevImage(&ptrDevB);
    destructHeatDevImage(&ptrDevH);

    delete[] ptrA;
    delete[] ptrB;
    delete[] ptrH;
    }

/*-------------------------*\
 |*     Methode override    *|
 \*-------------------------*/

void HeatTransfert::animationStep()
    {
this->t += 1;
    }

void HeatTransfert::process(uchar4* ptrDevPixels, int w, int h)
    {
    float *input, *output;

    input = isAInput ? ptrDevA : ptrDevB;
    output = isAInput ? ptrDevB : ptrDevA;

    launchHeatDiffusion(input, output, k, w, h);
    launchHeatEcrasement(output, ptrDevH, w, h);
    launchHeatDiffusion(output, input, k, w, h);
    launchHeatEcrasement(input, ptrDevH, w, h);

    launchHeatImageHSB(output, ptrDevPixels, calibreur, w, h);
    }

__host__ void HeatTransfert::instanciateHeatDevImage(float** ptr, float** ptrDev, size_t size)
    {
    HANDLE_ERROR(cudaMalloc((void** )ptrDev, size));
    HANDLE_ERROR(cudaMemcpy(*ptrDev, *ptr, size, cudaMemcpyHostToDevice));
    }

__host__ void HeatTransfert::destructHeatDevImage(float** ptrDev)
    {
    HANDLE_ERROR(cudaFree(*ptrDev));
    }

__host__ void HeatTransfert::launchHeatDiffusion(float* ptrDevIn, float* ptrDevOut, float k, int w, int h)
    {
    heatDiffusion<<<this->dg, this->db>>>(ptrDevIn, ptrDevOut, k, w, h);
    }

__host__ void HeatTransfert::launchHeatEcrasement(float* ptrDevCurrent, float* ptrDevH, int w, int h)
    {
    heatEcrasement<<<this->dg,this->db>>>(ptrDevCurrent, ptrDevH, w, h);
    }

__host__ void HeatTransfert::launchHeatImageHSB(float* ptrDevIn, uchar4* ptrDevOut, CalibreurF& calibreur, int w, int h)
    {
    heatImageHSB<<<this->dg,this->db>>>(ptrDevIn, ptrDevOut, calibreur, w, h);
    }

float HeatTransfert::getT(void)
    {
    return nbIteration;
    }

string HeatTransfert::getTitle(void)
    {
    return title;
    }
int HeatTransfert::getW(void)
    {
    return w;
    }
int HeatTransfert::getH(void)
    {
    return h;
    }
/**
 * Override
 */
float HeatTransfert::getAnimationPara(void)
{
   return t;
}
/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

