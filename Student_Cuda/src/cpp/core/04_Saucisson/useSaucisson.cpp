#include <iostream>
#include <limits.h>

using std::cout;
using std::endl;

extern bool saucissonSM(long);

bool useSaucisson(void)
    {
    cout << endl << "[SaucissonSM]" << endl;

    bool isOk = true;

    isOk&= saucissonSM(1e6);

    return true;
    }

