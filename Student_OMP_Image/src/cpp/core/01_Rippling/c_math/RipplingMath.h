#ifndef RIPPLING_MATH_H_
#define RIPPLING_MATH_H_

#include <cmath>
/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * Dans un header only pour preparer la version cuda
 */
class RipplingMath
    {

	/*--------------------------------------*\
	|*		Constructeur		*|
	 \*-------------------------------------*/

    public:

	RipplingMath(unsigned int w, unsigned int h)
	    {
	    this->dim2 = w / 2;
	    }

	virtual ~RipplingMath(void)
	    {
	    //rien
	    }

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	/**
	 * ptrColor represente le pixel (i,j) de l'image. uchar pour 4 cannaux color (r,g,b,alpha) chacun dans [0,255]
	 */
	void colorIJ(uchar4* ptrColor, int i, int j, float t)
	    {
		uchar levelGris;

		float dijResult;

		dij(i,j,&dijResult);
		levelGris = 128 + 127 * ((cos((dijResult/(10.0))-(t-7.0))) / ((dijResult/10.0)+1));

		ptrColor->x = levelGris;
		ptrColor->y = levelGris;
		ptrColor->z = levelGris;

		// ptrColor->w = 255; // alpha opaque
	    }

    private:

	void dij(int i, int j, float* ptrResult) // par exmple
	    {
	    //TODO
	     float fi = i - this->dim2;
	     float fj = j - this->dim2;
	     *ptrResult =  sqrt(fi*fi + fj*fj);
	    }


	/*--------------------------------------*\
	|*		Attribut		*|
	 \*-------------------------------------*/

    private:

	// Tools
	double dim2; //=dim/2

    };

#endif

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
