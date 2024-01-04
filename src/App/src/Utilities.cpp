#include "Utilities.h"

float operator-(timespec ts1, timespec ts2)
{
    time_t secDifPart = ts1.tv_sec - ts2.tv_sec;
    long nsecDifPart = ts1.tv_nsec - ts2.tv_nsec;
    float nsDif = static_cast<float>((secDifPart * 1000000000) + nsecDifPart);
    return nsDif;
}

