#include <iostream>
#include <limits.h>
#include "MathTools.h"

using std::cout;
using std::endl;

extern bool useMonteCarlo(long n);

bool useMontecarloMono(void)
    {
    cout << endl << "[MonteCarlo Mono]" << endl;

    bool isOk = true;

    isOk &= useMonteCarlo(1e11);

    return isOk;
    }
