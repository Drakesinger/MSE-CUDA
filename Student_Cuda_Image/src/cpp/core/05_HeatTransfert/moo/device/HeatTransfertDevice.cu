#include <iostream>

#include "Indice1D.h"
#include "Indice2D.h"
#include "cudaTools.h"
#include "Device.h"
#include "IndiceTools.h"
#include "DomaineMath.h"
#include "ImageFonctionel.h"
#include "CalibreurF.h"
#include "ColorTools.h"

using std::cout;
using std::endl;

__global__ void heatDiffusion(float* ptrDevIn, float* ptrDevOut, float k, int w, int h);
__global__ void heatEcrasement(float* ptrDevCurrent, float* ptrDevH, int w, int h);
__global__ void heatImageHSB(float* ptrDevIn, uchar4* ptrDevOut, CalibreurF calibreur, int w, int h);

__global__ void heatDiffusion(float* ptrDevIn, float* ptrDevOut, float k, int w, int h)
    {
    int tid = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int MAX = (w-2) * (h-2);

    int s = tid;
    while(s < MAX)
	{
	int C = s + w + 1; //Add the offset W & H
	ptrDevOut[C] = ptrDevIn[C] + k * (ptrDevIn[C - w] + ptrDevIn[C - 1] + ptrDevIn[C + w] + ptrDevIn[C + 1] - 4*ptrDevIn[C]);
	s += NB_THREAD;
	}
    }

__global__ void heatEcrasement(float* ptrDevCurrent, float* ptrDevH, int w, int h)
    {
    int tid = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int MAX = w * h;

    int s = tid;
    while(s < MAX)
	{
	if (ptrDevH[s] > 0)
	    ptrDevCurrent[s] = ptrDevH[s];
	s += NB_THREAD;
	}
    }

__global__ void heatImageHSB(float* ptrDevIn, uchar4* ptrDevOut, CalibreurF calibreur, int w, int h)
    {
    int tid = Indice2D::tid();
    const int NB_THREAD = Indice2D::nbThread();
    const int MAX = w * h;

    int s = tid;
    while(s < MAX)
	{
	ptrDevOut[s].w = 255;
	calibreur.calibrer(ptrDevIn[s]);
	ColorTools::HSB_TO_RVB(ptrDevIn[s], 1.0f, 1.0f, &ptrDevOut[s]);
	s += NB_THREAD;
	}
    }

