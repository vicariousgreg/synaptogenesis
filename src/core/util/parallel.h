#ifndef parallel_h
#define parallel_h

#include "util/constants.h"
#include "util/error_manager.h"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

// Define prefixes such that it doesn't affect anything for serial version
#ifdef __CUDACC__

#define GLOBAL __global__
#define DEVICE __device__
#define HOST __host__
#define device_check_error(msg) { gpuAssert(__FILE__, __LINE__, msg); }
#else

#define GLOBAL
#define DEVICE
#define HOST
#define device_check_error(msg)

// These will be dummy functions without CUDA
inline void device_synchronize() { }
inline int get_num_cuda_devices() { return 0; }
inline void init_rand(int count) { }
inline void free_rand() { }
inline int calc_threads(int computations) { return 0; }
inline int calc_blocks(int computations, int threads=0) { return 0; }
inline void device_check_memory(DeviceID device_id, size_t *free, size_t *total) { }
inline void* cuda_allocate_device(int device_id, size_t count,
        size_t size, void* source_data) {
    LOG_ERROR(
        "Attempted to allocate device memory in non-parallel build.");
    return nullptr;
}

#endif

#ifdef __CUDACC__

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

void device_synchronize();
int get_num_cuda_devices();

int calc_threads(int computations);
int calc_blocks(int computations, int threads=IDEAL_THREADS);

void gpuAssert(const char* file, int line, const char* msg);

/** Checks cuda memory usage and availability */
void device_check_memory(DeviceID device_id, size_t *free, size_t *total);
void* cuda_allocate_device(int device_id, size_t count,
        size_t size, void* source_data);

// Random state data
extern DEVICE curandState_t* cuda_rand_states;
GLOBAL void init_curand(int count);
void init_rand(int count);
void free_rand();

template<class T>
class Pointer;

template<class T>
GLOBAL void transpose_matrix_parallel(
    const Pointer<T> idata, Pointer<T> odata,
	const int original_rows, const int original_columns);

#endif

#endif
