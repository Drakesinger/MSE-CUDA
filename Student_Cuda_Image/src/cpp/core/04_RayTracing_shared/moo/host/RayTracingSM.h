#ifndef RAYTRACING_H_
#define RAYTRACING_H_

#include "cudaTools.h"
#include "Animable_I.h"
#include "MathTools.h"
#include "Sphere.h"
#include "IntervalF_CPU.h"
#include "VariateurF.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class RayTracingSM: public Animable_I
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	RayTracingSM(int w, int h);
	virtual ~RayTracingSM(void);

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

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;

	// Tools
	dim3 dg;
	dim3 db;
	float t;
	int nbSphere;
	Sphere* tabValue;



	//Outputs
	string title;

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
