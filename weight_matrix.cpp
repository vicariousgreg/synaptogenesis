#include <cstdlib>
#include "weight_matrix.h"
#include "layer.h"
#include "tools.h"

#ifdef parallel
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

WeightMatrix::WeightMatrix (Layer from_layer, Layer to_layer,
        bool plastic, float max_weight) :
            from_index(from_layer.start_index),
            to_index(to_layer.start_index),
            from_size(from_layer.size),
            to_size(to_layer.size),
            plastic(plastic),
            max_weight(max_weight),
            sign(from_layer.sign) { }

WeightMatrix::~WeightMatrix () {
#ifdef parallel
    cudaFree(this->mData);
#else
    free(this->mData);
#endif
}

void WeightMatrix::build() {
#ifdef parallel
    cudaMalloc(((void**)&this->mData), to_size * from_size * sizeof(float));
#else
    mData = (float*)malloc(to_size * from_size * sizeof(float));
#endif
    this->randomize(true, this->max_weight);
}

void WeightMatrix::randomize(bool self_connected, float max_weight) {
#ifdef parallel
    int matrix_size = this->from_size * this->to_size;
    float* temp_matrix = (float*)malloc(matrix_size * sizeof(float));
#endif
    for (int row = 0 ; row < this->from_size ; ++row) {
        for (int col = 0 ; col < this->to_size ; ++col) {
            if (self_connected || (row != col))
#ifdef parallel
                temp_matrix[row * to_size + col] = fRand(0, max_weight);
#else
                this->mData[row * to_size + col] = fRand(0, max_weight);
#endif
        }
    }

#ifdef parallel
    cudaMemcpy(this->mData, temp_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    free(temp_matrix);
#endif
}
