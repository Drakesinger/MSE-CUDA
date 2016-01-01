#include "Device.h"
#include <iostream>
#include "MathTools.h"
#include "cudaTools.h"
#include <algorithm>

__host__ static bool runGPU(int n);
__global__ static void kernelHistogramme(int* ptrDevInput, int* ptrDevOut, int n, int sizeHistogramme);

__host__ static int randomMinMax(int min, int max)
    {
    return (int)((max-min)*((float)(rand())/(float)RAND_MAX) + min);
    }

__host__ bool histogrammeGM(int n)
    {
    return runGPU(n);
    }

__host__ bool runGPU(int n)
    {
    int nMaxValue = 256;
    dim3 dg = dim3(nMaxValue, 1, 1);
    dim3 db = dim3(nMaxValue, 1, 1);

    size_t sizeHistogramme = db.x*sizeof(int);

    int* ptrInput = new int[n];
    int* ptrOut = new int[nMaxValue];
    for(int i = 0; i < n; ++i)
	ptrInput[i] = i%nMaxValue;

    for(int i = 0; i < nMaxValue; ++i)
	ptrOut[i] = 0;

    for(int i = 0; i < n; ++i)
	std::swap(ptrInput[randomMinMax(0, n-1)], ptrInput[randomMinMax(0, n-1)]);

    int* ptrDevInput;
    int* ptrDevOut;

    HANDLE_ERROR(cudaMalloc((void**)&ptrDevInput, n*sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(ptrDevInput, ptrInput, n*sizeof(int), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&ptrDevOut, sizeHistogramme));
    HANDLE_ERROR(cudaMemcpy(ptrDevOut, ptrOut, sizeHistogramme, cudaMemcpyHostToDevice));

    kernelHistogramme<<<dg, db, sizeHistogramme>>>(ptrDevInput, ptrDevOut, n, nMaxValue);

    HANDLE_ERROR(cudaMemcpy(ptrOut, ptrDevOut, sizeHistogramme, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(ptrDevOut));
    HANDLE_ERROR(cudaFree(ptrDevInput));

    bool isOk = true;
    for(int i = 0;isOk && i < nMaxValue-1; ++i)
	isOk &= ptrOut[i] == ptrOut[i+1];

    delete[] ptrInput;
    delete[] ptrOut;

    std::cout << "Histogramme GM : " << std::boolalpha << isOk << std::endl;

    return isOk;
    }

__global__ void kernelHistogramme(int* ptrDevInput, int* ptrDevOut, int n, int sizeHistogramme)
    {
    extern __shared__ int tabSM[];

    int tid = threadIdx.x + gridDim.x*blockIdx.x;
    const int NB_THREAD = gridDim.x*blockDim.x;
    int s = tid;

    if(threadIdx.x < sizeHistogramme)
	tabSM[threadIdx.x] = 0;

    __syncthreads();

    while(s < n)
	{
	atomicAdd(&tabSM[ptrDevInput[s]], 1);
	s += NB_THREAD;
	}

    __syncthreads();

    if(threadIdx.x < sizeHistogramme)
	atomicAdd(&ptrDevOut[threadIdx.x], tabSM[threadIdx.x]);
    }
