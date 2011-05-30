#ifndef __POISEUILEEFLOWUTIL_CU__
#define __POISEUILEEFLOWUTIL_CU__

__device__ int EvaluateShift(int pos, int size)
{
    if (pos < 0)
        return -1;
    if (pos > size - 1)
        return 1;
    return 0;
}

#endif//__POISEUILEEFLOWUTIL_CU__