#include <cstdlib>

#include "weight_matrix.h"
#include "tools.h"
#include "parallel.h"

WeightMatrix::WeightMatrix (Layer from_layer, Layer to_layer,
        bool plastic, float max_weight) :
            from_index(from_layer.start_index),
            to_index(to_layer.start_index),
            from_size(from_layer.size),
            to_size(to_layer.size),
            plastic(plastic),
            max_weight(max_weight),
            sign(from_layer.sign) { }

bool WeightMatrix::build() {
#ifdef PARALLEL
    cudaMalloc(((void**)&this->mData), to_size * from_size * sizeof(float));
    if (!cudaCheckError()) return false;
#else
    mData = (float*)malloc(to_size * from_size * sizeof(float));
    if (mData == NULL) return false;
#endif
    this->randomize(true, this->max_weight);
    return true;
}

void WeightMatrix::randomize(bool self_connected, float max_weight) {
#ifdef PARALLEL
    int matrix_size = this->from_size * this->to_size;
    float* temp_matrix = (float*)malloc(matrix_size * sizeof(float));
#endif
    for (int row = 0 ; row < this->from_size ; ++row) {
        for (int col = 0 ; col < this->to_size ; ++col) {
            if (self_connected || (row != col))
#ifdef PARALLEL
                temp_matrix[row * to_size + col] = fRand(0, max_weight);
#else
                this->mData[row * to_size + col] = fRand(0, max_weight);
#endif
        }
    }

#ifdef PARALLEL
    cudaDeviceSynchronize();
    cudaMemcpy(this->mData, temp_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(temp_matrix);
#endif
}
