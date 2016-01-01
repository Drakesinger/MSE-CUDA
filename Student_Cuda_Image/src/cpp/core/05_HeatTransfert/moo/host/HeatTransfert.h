#ifndef HEATTRANSFERT_H_
#define HEATTRANSFERT_H_

#include "cudaTools.h"
#include "Animable_I.h"
#include "MathTools.h"
#include "CalibreurF.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class HeatTransfert: public Animable_I
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	HeatTransfert(int w, int h, int blindIteration, float k);
	virtual ~HeatTransfert(void);

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*----------------*\
	|*  Override	  *|
	 \*---------------*/

	virtual void process(uchar4* ptrDevPixels, int w, int h);
	virtual void animationStep(void);
	virtual float getAnimationPara();

	virtual int getW(void);
	virtual int getH(void);
	virtual float getT(void);
	virtual string getTitle(void);

	__host__ void instanciateHeatDevImage(float**, float**, size_t);
	__host__ void destructHeatDevImage(float**);

	__host__ void launchHeatDiffusion(float* ptrDevInput, float* ptrDevOutput, float k, int w, int h);
	__host__ void launchHeatEcrasement(float* ptrDev, float* ptrDevH, int w, int h);
	__host__ void launchHeatImageHSB(float* ptrDevInput, uchar4* ptrDevOutput, CalibreurF& calibreur, int w, int h);

    private:

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;
	float k;
	float t;
	int nbBlindIteration;
	int nbIteration;
	bool isAInput;

	float* ptrA;
	float* ptrB;
	float* ptrH;

	float* ptrDevA;
	float* ptrDevB;
	float* ptrDevH;

	CalibreurF calibreur;

	// Tools
	dim3 dg;
	dim3 db;


	//Outputs
	string title;
    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
