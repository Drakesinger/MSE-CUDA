#ifndef MANDELBROTJULIA_MOO_H_
#define MANDELBROTJULIA_MOO_H_

#include "cudaType.h"
#include "AnimableFonctionel_I.h"

#include "VariateurI.h"
#include "Fractal.h" // car use dans .h
/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class MandelbrotJuliaMOO: public AnimableFonctionel_I
    {
	/*--------------------------------------*\
	 |*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	MandelbrotJuliaMOO(unsigned int w, unsigned int h, float dt, int n);
	virtual ~MandelbrotJuliaMOO(void);

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	/*-------------------------*\
	|*   Override Animable_I   *|
	 \*------------------------*/

	/**
	 * Call periodicly by the api
	 */
	virtual void process(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath);
	/**
	 * Call periodicly by the api
	 */
	virtual void animationStep();

	virtual float getAnimationPara();
	virtual int getW();
	virtual int getH();
	virtual string getTitle(void);
	virtual DomaineMath* getDomaineMathInit(void);

	virtual void setParallelPatern(ParallelPatern parallelPatern);

    private:

	void forAutoOMP(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath);
	void entrelacementOMP(uchar4* ptrTabPixels, int w, int h, const DomaineMath& domaineMath);

	void workPixel(uchar4* ptrColorIJ, int i, int j, int s, const DomaineMath& domaineMath, Fractal* ptrDamierMath);

	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

    protected:

	// Inputs
	int n;
	unsigned int w;
	unsigned int h;
	DomaineMath domaineMathInit;

	// Outputs
	string title;

	// Tools
	VariateurI variateurN; // fait varier para animation t
	ParallelPatern parallelPatern;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
