#include <iostream>
#include <limits.h>
#include "MathTools.h"

using std::cout;
using std::endl;

extern bool useMonteCarloMultiOMP(long n);

bool useMontecarloMulti(void)
    {
    cout << endl << "[MonteCarloMulti]" << endl;

    bool isOk = true;

    isOk &= useMonteCarloMultiOMP(1e11);

    return isOk;
    }
