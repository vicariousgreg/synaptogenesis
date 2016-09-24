#ifndef parallel_h
#define parallel_h

#ifdef PARALLEL

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

bool cudaCheckError() {
    cudaError_t e = cudaGetLastError();
    return (e != cudaSuccess);
}

void cudaSync() {
    cudaDeviceSynchronize();
}

#endif

#endif
