#include "iostream"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float div(int x, float f) {
    return float(x) / f;
}

template <typename OUT, typename... ARGS>
__global__ void sum(float(*func)(OUT, ARGS...), OUT* outputs, int size, float* dest, ARGS... args) {
    float s = 0.0;
    for (int i = 0 ; i < size ; ++i) {
        s += (*func)(outputs[i], args...);
    }
    *dest = s;
    //*dest = 0.0;
}

template <typename... Types>
__global__ void setup_kernel(float (**my_callback)(Types...)){
  *my_callback = &div;
}

void check_error() {
    cudaDeviceSynchronize();

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Cuda failure: '%s'\n", cudaGetErrorString(e));
        throw "";
    }
}

int main(void) {
    int vals[] = { 1, 2, 3 };
    int* device_vals;
    cudaMalloc(&device_vals, 3 * sizeof(int));
    cudaMemcpy(device_vals, vals, 3 * sizeof(int), cudaMemcpyHostToDevice);

    float(*local_div)(int, float);
    float(**device_div)(int, float);
    cudaMalloc(&device_div, sizeof(void *));

    setup_kernel<<<1, 1>>>((float(**)(int, float))device_div);
    cudaMemcpy((void *)&local_div, (void *)device_div, sizeof(void *), cudaMemcpyDeviceToHost);

    float *answer;
    cudaMalloc(&answer, sizeof(float));

    sum<<<1, 1>>>(local_div, device_vals, 3, answer, (float) 2.0);
    check_error();

    float local_answer = 1.0;
    cudaMemcpy(&local_answer, answer, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << local_answer;
}
