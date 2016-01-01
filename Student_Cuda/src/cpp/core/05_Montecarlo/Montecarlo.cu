#include "Device.h"
#include <iostream>
#include "MathTools.h"
#include "cudaTools.h"
#include "curand.h"
#include "curand_kernel.h"
#include <omp.h>
#include "Indice1D.h"

__host__ static float resultatTheorique();
__host__ static void runGPU(float*, long n);
__global__ static void kernelMonteCarlo_mono(curandState*, float* ptrDevOut, long n);
__global__ static void setup_kernel_rand_mono(curandState* tabGeneratorThread);
__device__ static int work(float x, float y, float dx);
__device__ static float f(float x);

__device__ static void reductionIntraBlock(int* tabSM, long n);
__device__ static void reductionInterBlock(int* tabSM, float*);

__device__ __host__ static float getXmin();
__device__ __host__ static float getXmax();
__device__ __host__ static float getYmin();
__device__ __host__ static float getYmax();

__host__ bool useMonteCarlo(long n)
    {
    float resEmpirique = 0;
    float resTheorique = resultatTheorique();
    runGPU(&resEmpirique, n);
    bool resultat = MathTools::isEquals(resTheorique, resEmpirique, (float) 1e-4);

    std::cout << "Résultat théorique : " << resTheorique << std::endl;
    std::cout << "Résultat empirique : " << resEmpirique << std::endl;
    std::cout << resultat << std::endl;

    return resultat;
    }

__host__ void runGPU(float* ptrOut, long n)
    {
    dim3 dg = dim3(256, 1, 1);
    dim3 db = dim3(256, 1, 1);
    Device::assertDim(dg, db);

    float result = 0;

    curandState* ptrDevGenerators;
    float* ptrDevOut;

    HANDLE_ERROR(cudaMalloc((void** )&ptrDevGenerators, db.x * sizeof(curandState*)));
    HANDLE_ERROR(cudaMalloc((void** )&ptrDevOut, sizeof(float)));
    HANDLE_ERROR(cudaMemset(ptrDevOut, 0, sizeof(float)));

    setup_kernel_rand_mono<<<dg, db>>>(ptrDevGenerators);

    kernelMonteCarlo_mono<<<dg, db, db.x*sizeof(int)>>>(ptrDevGenerators, ptrDevOut, n);

    HANDLE_ERROR(cudaMemcpy(&result, ptrDevOut, sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(ptrDevOut));
    HANDLE_ERROR(cudaFree(ptrDevGenerators));

    *ptrOut = result;

    *ptrOut = 2.0 * *ptrOut / (float) n * (getXmax() - getXmin()) * getYmax();
    }

__global__ void setup_kernel_rand_mono(curandState* tabGeneratorThread)
    {
    int tid = threadIdx.x;

    int deltaSeed = INT_MAX;
    int deltaSequence = 100;
    int deltaOffset = 100;

    int seed = 1234 + deltaSeed;
    int sequenceNumber = tid + deltaSequence;
    int offset = deltaOffset;

    curand_init(seed, sequenceNumber, offset, &tabGeneratorThread[tid]);
    }

__global__ void kernelMonteCarlo_mono(curandState* ptrDevGenerators, float* ptrDevOut, long n)
    {
    extern __shared__ int tabForBlock[];

    int tid = Indice1D::tid();
    const int NB_THREAD = Indice1D::nbThread();
    curandState localState = ptrDevGenerators[threadIdx.x];

    long s = tid;
    int sum = 0;
    float dx = (getXmax() - getXmin()) / (float) (NB_THREAD);
    while (s < n)
	{
	sum += work(curand_uniform(&localState), curand_uniform(&localState), dx);
	s += NB_THREAD;
	}

    tabForBlock[threadIdx.x] = sum;

    __syncthreads();

    reductionIntraBlock(tabForBlock, blockDim.x);
    reductionInterBlock(tabForBlock, ptrDevOut);
    }

__device__ void reductionIntraBlock(int* tabSM, long n)
    {
    long moitie = n / 2;
    while (moitie >= 1)
	{
	int tid = threadIdx.x;

	if (tid < moitie)
	    tabSM[tid] += tabSM[tid + moitie];
	moitie /= 2;

	__syncthreads();
	}
    }

__device__ void reductionInterBlock(int* tabSM, float* ptrDevOut)
    {
    if (threadIdx.x == 0)
	atomicAdd(ptrDevOut, tabSM[0]);
    }

__device__ int work(float x, float y, float dx)
    {
    float finalY = (getYmax() - getYmin()) * y + getYmin();

    float minX = getXmin() + dx * (threadIdx.x + gridDim.x * blockIdx.x);
    float maxX = minX + dx;
    float finalX = (maxX - minX) * dx + minX;

    return finalY <= f(finalX) ? 1 : 0;
    }

__host__ float resultatTheorique()
    {
    return 3.1415926535897932384626433832;
    }

__device__ __host__ float getXmin()
    {
    return -1.0;
    }

__device__ __host__ float getXmax()
    {
    return 1.0;
    }

__device__ __host__ float getYmin()
    {
    return 0.0;
    }

__device__ __host__ float getYmax()
    {
    return 1.0;
    }

__device__ float f(float x)
    {
    return sqrt(1 - x * x);
    }
