#ifndef parallel_h
#define parallel_h

// Define KERNEL prefix such that it doesn't affect anything for serial version
#ifdef PARALLEL
#define KERNEL __global__
#else
#define KERNEL
#endif


#ifdef PARALLEL

#include <cstdio>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assert.h"

#define THREADS 32

inline int calc_blocks(int computations) {
    return ceil((float) computations / THREADS);
}

inline int calc_blocks(int computations, int threads) {
    return ceil((float) computations / threads);
}

inline void cudaSync() {
    cudaDeviceSynchronize();
}

inline void cudaCheckError(const char* msg) {
    cudaSync();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Cuda failure: '%s'\n", cudaGetErrorString(e));
        throw msg;
    }
}

/** Checks cuda memory usage and availability */
inline void check_memory() {
    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
#endif

#endif
