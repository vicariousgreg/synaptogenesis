#include "util/parallel.h"
#include "util/resources/pointer.h"

#ifdef __CUDACC__

int calc_threads(int computations) {
    if (computations < IDEAL_THREADS) {
        int threads = 2;
        while (threads < computations)
            threads *= 2;
        return threads;
    } else {
        return IDEAL_THREADS;
    }
}

int calc_blocks(int computations, int threads) {
    if (threads == 0)
        threads = calc_threads(computations);
    return ceil((float) computations / threads);
}

void device_synchronize() {
    for (int i = 0; i < get_num_cuda_devices(); ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
}

int get_num_cuda_devices() {
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  return nDevices;
}

void gpuAssert(const char* file, int line, const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Cuda failure (%s: %d): '%s'\n", file, line,
            cudaGetErrorString(e));
        if (msg == nullptr) msg = "";
        LOG_ERROR(msg);
    }
}

void device_check_memory(DeviceID device_id, size_t *free, size_t *total) {
    if (device_id >= get_num_cuda_devices())
        LOG_ERROR("Tried to query invalid device memory!");

    int prev_device;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id);

    cudaMemGetInfo(free, total);

    cudaSetDevice(prev_device);
}

void* cuda_allocate_device(int device_id, size_t count,
        size_t size, void* source_data) {
    int prev_device;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id);

    void* ptr;
    cudaMalloc(&ptr, count * size);
    device_synchronize();
    device_check_error("Failed to allocate memory on device for neuron state!");
    if (source_data != nullptr)
        cudaMemcpy(ptr, source_data, count * size, cudaMemcpyHostToDevice);
    device_synchronize();
    device_check_error("Failed to initialize memory on device for neuron state!");

    cudaSetDevice(prev_device);
    return ptr;
}

DEVICE curandState_t* cuda_rand_states = nullptr;

GLOBAL void init_curand(int count){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count)
        curand_init(clock64(), idx, 0, &cuda_rand_states[idx]);
}

void init_rand(int count) {
    int prev_device;
    cudaGetDevice(&prev_device);
    for (int i = 0; i < get_num_cuda_devices(); ++i) {
        cudaSetDevice(i);
        curandState_t* states;
        cudaMalloc((void**) &states, count * sizeof(curandState_t));
        cudaMemcpyToSymbol(cuda_rand_states, &states, sizeof(void *));
        init_curand
            <<<calc_blocks(count), calc_threads(count)>>>(count);
    }
    cudaSetDevice(prev_device);
}

void free_rand() {
    int prev_device;
    cudaGetDevice(&prev_device);
    for (int i = 0; i < get_num_cuda_devices(); ++i) {
        cudaSetDevice(i);
        curandState_t* states;
        cudaMemcpyFromSymbol(&states, cuda_rand_states, sizeof(void *));
        cudaFree(states);

        states = nullptr;
        cudaMemcpyToSymbol(cuda_rand_states, &states, sizeof(void *));
    }
    cudaSetDevice(prev_device);
}

#endif
