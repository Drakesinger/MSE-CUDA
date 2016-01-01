#include "Device.h"
#include <iostream>
#include "MathTools.h"
#include "cudaTools.h"

__host__ static float resultatTheorique();
__host__ static void runGPU(float* ptrOut, long n);
__global__ static void kernelSaucissonSM(float* ptrDevOut, long n, float dx);
__device__ static float work(long, float);
__device__ static float fpi(float x);

__device__ static void reductionIntraBlock(float* tabSM, long n);
__device__ static void reductionInterBlock(float* tabSM, float*);

__host__ bool saucissonSM(long n)
    {
    float resEmpirique = 0;
    float resTheorique = resultatTheorique();
    runGPU(&resEmpirique, n);
    bool resultat = MathTools::isEquals(resTheorique, resEmpirique, (float)1e-4);

    std::cout << "Résultat théorique : " << resTheorique << std::endl;
    std::cout << "Résultat empirique : " << resEmpirique << std::endl;
    std::cout << std::boolalpha <<  resultat << std::endl;

    return resultat;
    }

__host__ void runGPU(float* ptrOut, long n)
    {
    dim3 dg = dim3(256, 1, 1);
    dim3 db = dim3(256, 1, 1);

    size_t sizeBlock = db.x*sizeof(float);
    Device::assertDim(dg, db);

    float* ptrDevOut;
    float dx = 1.0/n;

    HANDLE_ERROR(cudaMalloc((void**)&ptrDevOut, sizeof(float)));
    HANDLE_ERROR(cudaMemset(ptrDevOut, 0, sizeof(float)));

    kernelSaucissonSM<<<dg, db, sizeBlock>>>(ptrDevOut, n, dx);

    HANDLE_ERROR(cudaMemcpy(ptrOut, ptrDevOut, sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptrDevOut));

    *ptrOut /= n;
    }

__global__ void kernelSaucissonSM(float* ptrDevOut, long n, float dx)
    {
    extern __shared__ float tabForBlock[];

    int tid = threadIdx.x + gridDim.x*blockIdx.x;
    const int NB_THREAD = gridDim.x*blockDim.x;

    long s = tid;
    float sum = 0;
    while(s < n)
	{
	sum += work(s, dx);
	s += NB_THREAD;
	}

    tabForBlock[threadIdx.x] = sum;

    __syncthreads();

    reductionIntraBlock(tabForBlock, blockDim.x);
    reductionInterBlock(tabForBlock, ptrDevOut);
    }

__device__ void reductionIntraBlock(float* tabSM, long n)
    {
    long moitie = n/2;
    while(moitie >= 1)
	{
	int tid = threadIdx.x;

	if(tid < moitie)
	    tabSM[tid] += tabSM[tid+moitie];
	moitie /= 2;

	__syncthreads();
	}
    }

__device__ void reductionInterBlock(float* tabSM, float* ptrDevOut)
    {
    if(threadIdx.x == 0)
	atomicAdd(ptrDevOut, tabSM[0]);
    }

__device__ float work(long i, float dx)
    {
    return fpi(i*dx);
    }

__device__ float fpi(float x)
    {
    return 4 / (1 + x * x);
    }

__host__ float resultatTheorique()
    {
    return 3.1415926535897932384626433832;
    }
