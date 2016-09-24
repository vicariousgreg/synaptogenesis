#include <cstdlib>

#include "weight_matrix.h"
#include "tools.h"
#include "parallel.h"
#include "layer.h"

WeightMatrix::WeightMatrix (Layer &from_layer, Layer &to_layer,
        bool plastic, float max_weight) :
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(plastic),
            max_weight(max_weight),
            sign(from_layer.sign) { }

bool WeightMatrix::build() {
#ifdef PARALLEL
    cudaMalloc((&this->mData), to_layer.size * from_layer.size * sizeof(float));
    if (!cudaCheckError()) return false;
#else
    mData = (float*)malloc(to_layer.size * from_layer.size * sizeof(float));
    if (mData == NULL) return false;
#endif
    return this->randomize(true, this->max_weight);
}

bool WeightMatrix::randomize(bool self_connected, float max_weight) {
#ifdef PARALLEL
    int matrix_size = this->from_layer.size * this->to_layer.size;
    float* temp_matrix = (float*)malloc(matrix_size * sizeof(float));
    if (!temp_matrix) {
        printf("Failed to allocate temporary matrix on host!\n");
        return false;
    }
#endif
    for (int row = 0 ; row < this->from_layer.size ; ++row) {
        for (int col = 0 ; col < this->to_layer.size ; ++col) {
            if (self_connected || (row != col))
#ifdef PARALLEL
                temp_matrix[row * to_layer.size + col] = fRand(0, max_weight);
#else
                this->mData[row * to_layer.size + col] = fRand(0, max_weight);
#endif
        }
    }

#ifdef PARALLEL
    cudaMemcpy(this->mData, temp_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    bool success =  cudaCheckError();
    free(temp_matrix);
    return success;
#else
    return true;
#endif
}
