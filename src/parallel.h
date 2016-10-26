#ifndef parallel_h
#define parallel_h

// Define KERNEL prefix such that it doesn't affect anything for serial version
#ifdef PARALLEL
#define GLOBAL __global__
#define DEVICE __device__
#else
#define GLOBAL
#define DEVICE
#endif


#ifdef PARALLEL

#include <cstdio>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assert.h"

#include "error_manager.h"

#define WARP_SIZE 32
#define IDEAL_THREADS 128
#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

inline int calc_threads(int computations) {
    return IDEAL_THREADS;
}

inline int calc_blocks(int computations, int threads=IDEAL_THREADS) {
    return ceil((float) computations / calc_threads(computations));
}

inline void cudaSync() {
    cudaDeviceSynchronize();
}

#define cudaCheckError(msg) { gpuAssert(__FILE__, __LINE__, msg); }

inline void gpuAssert(const char* file, int line, const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Cuda failure (%s: %d): '%s'\n", file, line, cudaGetErrorString(e));
        ErrorManager::get_instance()->log_error(msg);
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
