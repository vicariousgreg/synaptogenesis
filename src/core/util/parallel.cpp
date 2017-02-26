#ifdef __CUDACC__

#include "util/parallel.h"

int calc_threads(int computations) {
    return IDEAL_THREADS;
}

int calc_blocks(int computations, int threads) {
    return ceil((float) computations / calc_threads(computations));
}

void cudaSync() {
    cudaDeviceSynchronize();
}

void gpuAssert(const char* file, int line, const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Cuda failure (%s: %d): '%s'\n", file, line, cudaGetErrorString(e));
        ErrorManager::get_instance()->log_error(msg);
    }
}

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

void* allocate_device(int count, int size, void* source_data) {
    void* ptr;
    cudaMalloc(&ptr, count * size);
    cudaCheckError("Failed to allocate memory on device for neuron state!");
    if (source_data != NULL)
        cudaMemcpy(ptr, source_data, count * size, cudaMemcpyHostToDevice);
    cudaCheckError("Failed to initialize memory on device for neuron state!");
    return ptr;
}

DEVICE curandState_t* cuda_rand_states = NULL;

GLOBAL void init_curand(int count){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count)
        curand_init(clock64(), idx, 0, &cuda_rand_states[idx]);
}

void init_cuda_rand(int count) {
    curandState_t* states;
    cudaMalloc((void**) &states, count * sizeof(curandState_t));
    cudaMemcpyToSymbol(cuda_rand_states, &states, sizeof(void *));
    init_curand
        <<<calc_blocks(count), calc_threads(count)>>>(count);
}

void free_cuda_rand() {
    curandState_t* states;
    cudaMemcpyFromSymbol(&states, cuda_rand_states, sizeof(void *));
    cudaFree(states);

    states = NULL;
    cudaMemcpyToSymbol(cuda_rand_states, &states, sizeof(void *));
}

#endif
