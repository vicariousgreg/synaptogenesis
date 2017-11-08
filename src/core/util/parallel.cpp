#include "util/parallel.h"

Memstat::Memstat(DeviceID device_id, size_t free, size_t total,
    size_t used, size_t used_by_this)
        : device_id(device_id), free(free), total(total),
          used(used), used_by_this(used_by_this) { }

Memstat::Memstat(const Memstat& o, size_t used_by_this)
        : device_id(o.device_id), free(o.free), total(o.total),
          used(o.used), used_by_this(used_by_this) { }

void Memstat::print() {
    if (free == 0)
        printf("Device %d memory usage:\n"
            "  used  : %10zu %7.2f MB\n",
            device_id,
            used_by_this, float(used_by_this)/1024.0/1024.0);
    else if (used_by_this > 0)
        printf("Device %d memory usage:\n"
            "  proc  : %10zu %7.2f MB\n"
            "  used  : %10zu %7.2f MB\n"
            "  free  : %10zu %7.2f MB\n"
            "  total : %10zu %7.2f MB\n",
            device_id,
            used_by_this, float(used_by_this)/1024.0/1024.0,
            used, float(used)/1024.0/1024.0,
            free, float(free)/1024.0/1024.0,
            total, float(total)/1024.0/1024.0);
    else
        printf("Device %d memory usage:\n"
            "  used  : %10zu %7.2f MB\n"
            "  free  : %10zu %7.2f MB\n"
            "  total : %10zu %7.2f MB\n",
            device_id,
            used, float(used)/1024.0/1024.0,
            free, float(free)/1024.0/1024.0,
            total, float(total)/1024.0/1024.0);
}

#ifdef __CUDACC__

int calc_threads(int computations) {
    return IDEAL_THREADS;
}

int calc_blocks(int computations, int threads) {
    return ceil((float) computations / calc_threads(computations));
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

Memstat device_check_memory(DeviceID device_id) {
    int prev_device;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id);

    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte );
    size_t used_byte = total_byte - free_byte;

    Memstat stats = Memstat(device_id, free_byte, total_byte, used_byte);
    cudaSetDevice(prev_device);
    return stats;
}

void* cuda_allocate_device(int device_id, size_t count,
        size_t size, void* source_data) {
    int prev_device;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id);

    void* ptr;
    cudaMalloc(&ptr, count * size);
    device_check_error("Failed to allocate memory on device for neuron state!");
    if (source_data != nullptr)
        cudaMemcpy(ptr, source_data, count * size, cudaMemcpyHostToDevice);
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
