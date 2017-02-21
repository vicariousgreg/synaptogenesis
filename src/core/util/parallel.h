#ifndef parallel_h
#define parallel_h

#include "constants.h"

// Define prefixes such that it doesn't affect anything for serial version
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
#include "curand.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include "assert.h"

#include "util/error_manager.h"

#define WARP_SIZE 32
#define IDEAL_THREADS 128
#define MAX_THREADS 1024
#define MAX_BLOCKS 65535

int calc_threads(int computations);
int calc_blocks(int computations, int threads=IDEAL_THREADS);

void cudaSync();

#define cudaCheckError(msg) { gpuAssert(__FILE__, __LINE__, msg); }
void gpuAssert(const char* file, int line, const char* msg);

/** Checks cuda memory usage and availability */
void check_memory();
void* allocate_device(int count, int size, void* source_data);

// Random state data
extern DEVICE curandState_t* cuda_rand_states;
GLOBAL void init_curand(int count);
void init_cuda_rand(int count);
void free_cuda_rand();

#endif

#endif
