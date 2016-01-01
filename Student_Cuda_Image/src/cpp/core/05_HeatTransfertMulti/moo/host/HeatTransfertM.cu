#include "HeatTransfertM.h"

#include <assert.h>
#include <stdio.h>
#include <iostream>

#include "Device.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern __global__ void diffusion(float* ptrImageInput, float* ptrImageOutput, int w, int h);
extern __global__ void ecrasement(float* ptrImageInOutput, float* ptrImageHeater, float* ptrImageOutput, int w, int h);
extern __global__ void toScreenImageHSB(uchar4* ptrDevPixels, float* ptrImageInput, int w, int h);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*-------------------------*\
 |*	Constructeur	    *|
 \*-------------------------*/

HeatTransfertM::HeatTransfertM(int w, int h, float dt)
    {
    // Inputs
    this->w = w;
    this->h = h;
    this->dt = dt;

    // Tools
    this->dg = dim3(16, 2, 1);
    this->db = dim3(32, 4, 1);
    this->t = 0;
    this->isImageAInput = true;
    this->NB_ITERATION_AVEUGLE = 10;
    this->nbIterations = 0;
    this->masterDeviceId = 0;

    // Outputs
    this->title = "HeatTransfert MultiGPU";

    this->nbGPU = 1;
    printf("NB GPU used : %d (must be peer-to-peer compatible GPUs !)\n", this->nbGPU);

    this->ptrDevImageHeater = new float*[this->nbGPU];
    this->ptrDevImageInput = new float*[this->nbGPU];
    this->ptrDevImageOutput = new float*[this->nbGPU];
    //this->stream = new cudaStream_t[this->nbGPU];

    createHeater();
    createImageInit();

    int currentDevice = Device::getDeviceId();
    for (int deviceId = 0; deviceId < this->nbGPU; deviceId++)
	{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceId);
	printf("Device id : %d, name : %s, major : %d, minor : %d\n", deviceId, props.name, props.major, props.minor);

	memoryManagment(deviceId);
	}
    HANDLE_ERROR(cudaSetDevice(currentDevice));

    //print(dg, db);
    Device::assertDim(dg, db);
    }

HeatTransfertM::~HeatTransfertM()
    {

    delete[] imageHeater;
    delete[] imageInit;

    for (int deviceId = 0; deviceId < this->nbGPU; deviceId++)
	{
	HANDLE_ERROR(cudaFree(ptrDevImageHeater[deviceId]));
	HANDLE_ERROR(cudaFree(ptrDevImageInput[deviceId]));
	HANDLE_ERROR(cudaFree(ptrDevImageOutput[deviceId]));

	ptrDevImageHeater[deviceId] = NULL;
	ptrDevImageInput[deviceId] = NULL;
	ptrDevImageOutput[deviceId] = NULL;
	}

    delete[] ptrDevImageHeater;
    delete[] ptrDevImageInput;
    delete[] ptrDevImageOutput;

    	HANDLE_ERROR(cudaFree(ptrDevImageA));
    	HANDLE_ERROR(cudaFree(ptrDevImageB));

    	ptrDevImageA = NULL;
    	ptrDevImageB = NULL;

    }

/*-------------------------*\
 |*	Methode		    *|
 \*-------------------------*/

void HeatTransfertM::memoryManagment(int deviceId)
    {
    HANDLE_ERROR(cudaSetDevice(deviceId));

    if (deviceId != masterDeviceId)
	{
	HANDLE_ERROR(cudaDeviceEnablePeerAccess(masterDeviceId, 0));
	}
    else
	{
	int size = IMAGEWIDTH * IMAGEHEIGHT * sizeof(float);

	this->ptrDevImageA = NULL;
	this->ptrDevImageB = NULL;

	HANDLE_ERROR(cudaMalloc(&ptrDevImageA, size));
	HANDLE_ERROR(cudaMalloc(&ptrDevImageB, size));

	HANDLE_ERROR(cudaMemset(ptrDevImageA, 0, size));
	HANDLE_ERROR(cudaMemset(ptrDevImageB, 0, size));

	HANDLE_ERROR(cudaMemcpy(ptrDevImageA, this->imageInit, size, cudaMemcpyHostToDevice));
	}

    this->ptrDevImageHeater[deviceId] = NULL;
    this->ptrDevImageInput[deviceId] = NULL;
    this->ptrDevImageOutput[deviceId] = NULL;

    int sizeChunk = getChunkSize(deviceId) * sizeof(float);
    int startIndex = getChunkStartIndex(deviceId);

    HANDLE_ERROR(cudaMalloc(&ptrDevImageHeater[deviceId], sizeChunk));
    HANDLE_ERROR(cudaMalloc(&ptrDevImageInput[deviceId], sizeChunk));
    HANDLE_ERROR(cudaMalloc(&ptrDevImageOutput[deviceId], sizeChunk));

    HANDLE_ERROR(cudaMemset(ptrDevImageHeater[deviceId], 0, sizeChunk));
    HANDLE_ERROR(cudaMemset(ptrDevImageInput[deviceId], 0, sizeChunk));
    HANDLE_ERROR(cudaMemset(ptrDevImageOutput[deviceId], 0, sizeChunk));

    HANDLE_ERROR(cudaMemcpy(ptrDevImageHeater[deviceId], &this->imageHeater[startIndex], sizeChunk, cudaMemcpyHostToDevice));

    }

void HeatTransfertM::createHeater()
    {
    imageHeater = new float[IMAGEWIDTH * IMAGEHEIGHT];



    for (int i = 0; i < IMAGEWIDTH; i++)
    {
	for (int j = 0; j < IMAGEHEIGHT; j++)
	{
	    //Carré chaud au milieu
	    if (i > IMAGEWIDTH/2 - 30 && i < IMAGEWIDTH/2 + 30 && j > IMAGEHEIGHT/2 - 30 && j < IMAGEHEIGHT/2 + 30)
		imageHeater[i * IMAGEWIDTH + j] = 1.0;
	    else
		imageHeater[i * IMAGEWIDTH + j] = 0.0;
	}
    }
    }

void HeatTransfertM::createImageInit()
    {
    imageInit = new float[IMAGEWIDTH * IMAGEHEIGHT];

    for (int i = 0; i < IMAGEWIDTH; i++)
	{
	for (int j = 0; j < IMAGEHEIGHT; j++)
	    {
	    imageInit[i * IMAGEWIDTH + j] = 0.0;
	    }
	}
    }

/**
 * Override
 */
void HeatTransfertM::process(uchar4* ptrDevPixels, int w, int h)
    {

    int currentDevice = Device::getDeviceId();

    float* ptrImageInput = NULL;
    float* ptrImageOutput = NULL;

    if (this->isImageAInput)
	{
	ptrImageInput = ptrDevImageA;
	ptrImageOutput = ptrDevImageB;
	}
    else
	{
	ptrImageInput = ptrDevImageB;
	ptrImageOutput = ptrDevImageA;
	}

//#pragma omp parallel for
    for (int deviceId = 0; deviceId < this->nbGPU; deviceId++)
	{
	HANDLE_ERROR(cudaSetDevice(deviceId));

	float* ptrMasterImageInputChunk = getImageChunk(ptrImageInput, deviceId); //get the address on the master GPU
	float* ptrMasterImageOutputChunk = getImageChunkWithoutBorder(ptrImageOutput, deviceId);

	diffusion<<<this->dg, this->db>>>(ptrMasterImageInputChunk, ptrDevImageOutput[deviceId], IMAGEWIDTH, getChunkHeight(deviceId));

	float* ptrDevImageOutputWithoutUpperBorder = ptrDevImageOutput[deviceId];
	if (deviceId > 0)
	    ptrDevImageOutputWithoutUpperBorder = &ptrDevImageOutputWithoutUpperBorder[IMAGEWIDTH];

    ecrasement<<<this->dg, this->db>>>(ptrDevImageOutputWithoutUpperBorder, ptrDevImageHeater[deviceId], ptrMasterImageOutputChunk, IMAGEWIDTH, getChunkHeightWithoutBorder());

    }

HANDLE_ERROR(cudaSetDevice(masterDeviceId));
HANDLE_ERROR(cudaDeviceSynchronize());

//Display the image

if(nbIterations % NB_ITERATION_AVEUGLE == 0)
	toScreenImageHSB<<<this->dg, this->db>>>(ptrDevPixels, ptrImageOutput, IMAGEWIDTH, IMAGEHEIGHT);

    HANDLE_ERROR(cudaSetDevice(currentDevice));

//printf("end process\n");

isImageAInput = !isImageAInput;
nbIterations++;

}

float* HeatTransfertM::getImageChunk(float* ptrImage, int deviceId)
{
int startIndex = getChunkStartIndex(deviceId);
return &ptrImage[startIndex];
}

float* HeatTransfertM::getImageChunkWithoutBorder(float* ptrImage, int deviceId)
{
int startIndex = deviceId * (IMAGEWIDTH * IMAGEHEIGHT / this->nbGPU);
return &ptrImage[startIndex];
}

int HeatTransfertM::getChunkHeight(int deviceId)
{
int bordure = 1;
if (this->nbGPU == 1)
    return getChunkHeightWithoutBorder();

else if (deviceId > 0 && deviceId < this->nbGPU - 1)
    {
    bordure = 2;
    }

return (IMAGEHEIGHT / this->nbGPU) + bordure;
}

int HeatTransfertM::getChunkHeightWithoutBorder()
{

return (IMAGEHEIGHT / this->nbGPU);
}

int HeatTransfertM::getChunkStartIndex(int deviceId)
{
int offset = 0;

if (deviceId > 0)
    {
    offset = -1;
    }

return (IMAGEWIDTH * IMAGEHEIGHT / this->nbGPU) * deviceId + offset * IMAGEWIDTH;
}

int HeatTransfertM::getChunkSize(int deviceId)
{
if (this->nbGPU == 1)
    return getChunkSizeWithoutBorder();

int bordure = 1;

if (deviceId > 0 && deviceId < this->nbGPU - 1)
    {
    bordure = 2;
    }

return (IMAGEWIDTH * IMAGEHEIGHT / this->nbGPU) + bordure * IMAGEWIDTH;
}

int HeatTransfertM::getChunkSizeWithoutBorder()
{
return IMAGEWIDTH * IMAGEHEIGHT / this->nbGPU;
}

/**
 * Override
 */
void HeatTransfertM::animationStep()
{
t += dt;
}

/*--------------*\
 |*	get	 *|
 \*--------------*/

/**
 * Override
 */
float HeatTransfertM::getAnimationPara(void)
{
return t;
}

/**
 * Override
 */
int HeatTransfertM::getW(void)
{
return w;
}

/**
 * Override
 */
int HeatTransfertM::getH(void)
{
return h;
}

/**
 * Override
 */
string HeatTransfertM::getTitle(void)
{
return title;
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

