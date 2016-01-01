#ifndef HEATTRANSFERTM_H_
#define HEATTRANSFERTM_H_

#include "cudaTools.h"
#include "Animable_I.h"
#include "MathTools.h"

#define IMAGEWIDTH 300
#define IMAGEHEIGHT 300

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class HeatTransfertM: public Animable_I
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	HeatTransfertM(int w, int h, float dt);
	virtual ~HeatTransfertM(void);

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*-------------------------*\
	|*   Override Animable_I   *|
	 \*------------------------*/

	/**
	 * Call periodicly by the api
	 */
	virtual void process(uchar4* ptrDevPixels, int w, int h);
	/**
	 * Call periodicly by the api
	 */
	virtual void animationStep();

	virtual float getAnimationPara();
	virtual string getTitle();
	virtual int getW();
	virtual int getH();

    private:

	void createHeater();
	void createImageInit();
	void memoryManagment(int deviceId);
	float* getImageChunk(float* ptrImage, int deviceId);
	float* getImageChunkWithoutBorder(float* ptrImage, int deviceId);
	int getChunkHeight(int deviceId);
	int getChunkHeightWithoutBorder();
	int getChunkStartIndex(int deviceId);
	int getChunkSize(int deviceId);
	int getChunkSizeWithoutBorder();


	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;
	float dt;

	// Tools
	dim3 dg;
	dim3 db;
	float t;
	bool isImageAInput;
	int nbGPU;
	int nbIterations;
	int NB_ITERATION_AVEUGLE;
	cudaStream_t* streams;
	int masterDeviceId;

	float* imageHeater;
	float* imageInit;
	float** ptrDevImageHeater;
	float** ptrDevImageInput;
	float** ptrDevImageOutput;
	float* ptrDevImageA;
	float* ptrDevImageB;

	//cudaStream_t* stream;

	//Outputs
	string title;
    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
