#include "util/parallel.h"
#include "util/pointer.h"

dim3 calc_transpose_threads(int original_rows, int original_columns) {
    return dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS, 1);
}

dim3 calc_transpose_blocks(int original_rows, int original_columns) {
    return dim3(
        (original_columns/TRANSPOSE_TILE_DIM)
            + (original_columns % TRANSPOSE_TILE_DIM > 0),
        (original_rows/TRANSPOSE_TILE_DIM)
            + (original_rows % TRANSPOSE_TILE_DIM > 0), 1);
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

template GLOBAL void transpose_matrix_parallel<float>(
	const Pointer<float> idata, Pointer<float> odata,
	const int original_rows, const int original_columns);
template GLOBAL void transpose_matrix_parallel<int>(
	const Pointer<int> idata, Pointer<int> odata,
	const int original_rows, const int original_columns);

template<class T>
GLOBAL void transpose_matrix_parallel(
        const Pointer<T> idata, Pointer<T> odata,
        const int original_rows, const int original_columns) {
    __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    T* in = idata.get();
    if (x < original_columns)
        for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
            if (y+j < original_rows)
                tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*original_columns + x];

    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    T* out = odata.get();
    if (x < original_rows)
        for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
            if (y + j < original_columns)
                out[(y+j)*original_rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

#endif
