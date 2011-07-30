#ifndef PERISTALSIS_UTIL_CU__
#define PERISTALSIS_UTIL_CU__

__device__ int EvaluateShift(int pos, int size)
{
    if (pos < 0)
        return -1;
    if (pos > size - 1)
        return 1;
    return 0;
}

#endif//PERISTALSIS_UTIL_CU__