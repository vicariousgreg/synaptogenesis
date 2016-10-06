#include "iostream"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float divide(int x, float f) {
    return float(x) / f;
}

__device__ float (*divide_ptr)(int, float) = divide;

__device__ float mult(int x, float f) {
    return x * f;
}

__device__ float (*mult_ptr)(int, float) = mult;


template <typename OUT, typename... ARGS>
__global__ void sum(float(*func)(OUT, ARGS...), OUT* outputs, int size, float* dest, ARGS... args) {
    float s = 0.0;
    for (int i = 0 ; i < size ; ++i) {
        s += (*func)(outputs[i], args...);
    }
    *dest = s;
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
    // Set up testing data
    int vals[] = { 1, 2, 3 };
    int* device_vals;
    cudaMalloc(&device_vals, 3 * sizeof(int));
    cudaMemcpy(device_vals, vals, 3 * sizeof(int), cudaMemcpyHostToDevice);

    // Set up space for the result
    float local_answer = 0.0;
    float *device_answer;
    cudaMalloc(&device_answer, sizeof(float));

    // DIVIDE
    float(*local_div)(int, float);
    cudaMemcpyFromSymbol(&local_div, divide_ptr, sizeof(void *));
    sum<int, float><<<1, 1>>>(local_div, device_vals, 3, device_answer, (float) 2.0);
    cudaMemcpy(&local_answer, device_answer, sizeof(float), cudaMemcpyDeviceToHost);
    check_error();
    std::cout << local_answer << "\n";

    // MULT
    float(*local_mult)(int, float);
    cudaMemcpyFromSymbol(&local_mult, mult_ptr, sizeof(void *));
    sum<int, float><<<1, 1>>>(local_mult, device_vals, 3, device_answer, (float) 2.0);
    cudaMemcpy(&local_answer, device_answer, sizeof(float), cudaMemcpyDeviceToHost);
    check_error();
    std::cout << local_answer << "\n";
}
