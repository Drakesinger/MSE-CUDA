#include "Indice2D.h"
#include "cudaTools.h"
#include "Device.h"

#include "RayTracingSMMath.h"
#include "IndiceTools.h"
#include "Sphere.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

__global__ void raytracingSM(uchar4* ptrDevPixels, Sphere* ptrDevSphere, int nbSphere, int w, int h, float t);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static __device__ void GMtoSM(Sphere* tabSphereSM,Sphere* tabSphereGM,int nbSphere);
static __device__ void work(uchar4* ptrDevPixels,Sphere* ptrDevSphere,int nbSphere,int w, int h, float t);
/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__global__ void raytracingSM(uchar4* ptrDevPixels, Sphere* ptrDevSphereGM, int nbSphere, int w, int h, float t)
    {
    extern __shared__ Sphere tabSphereSM[];
    GMtoSM(tabSphereSM,ptrDevSphereGM,nbSphere);
    __syncthreads;
    work(ptrDevPixels,tabSphereSM,nbSphere,w,h,t);
    }

__device__ void work(uchar4* ptrDevPixels,Sphere* ptrDevSphere,int nbSphere,int w, int h, float t)
    {
    RayTracingSMMath rayTracingSMMath = RayTracingSMMath(t); // ici pour preparer cuda

    const int WH = w * h;

    const int NB_THREAD = Indice2D::nbThread();
    const int TID = Indice2D::tid();

    int s = TID;

    int i;
    int j;
    while (s < WH)
	{
	IndiceTools::toIJ(s, w, &i, &j);
	rayTracingSMMath.colorXY(&ptrDevPixels[s],ptrDevSphere,nbSphere, i,j,t);
	s += NB_THREAD;
	}
    }

__device__ void GMtoSM(Sphere* tabSphereSM,Sphere* tabSphereGM,int nbSphere)
    {
    int s = Indice2D::tidLocal();
    const int NBTHREAD_BLOCK= Indice2D::nbThreadBlock();
    while(s<nbSphere)
	{
	tabSphereSM[s]=tabSphereGM[s];
	s+=NBTHREAD_BLOCK;
	}

    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

