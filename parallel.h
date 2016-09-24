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

/** Checks cuda memory usage and availability */
void check_memory() {
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
