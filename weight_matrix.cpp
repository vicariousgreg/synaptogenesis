#include <cstdlib>

#include "weight_matrix.h"
#include "tools.h"
#include "parallel.h"
#include "layer.h"

WeightMatrix::WeightMatrix (Layer &from_layer, Layer &to_layer,
        bool plastic, float max_weight, MatrixType type) :
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(plastic),
            max_weight(max_weight),
            sign(from_layer.sign),
            type(type) {
    if (type == FULLY_CONNECTED) {
        this->matrix_size = from_layer.size * to_layer.size;
    } else if (type == ONE_TO_ONE) {
        if (from_layer.size != to_layer.size) {
            throw;
        } else {
            this->matrix_size = from_layer.size;
        }
    }
}

void WeightMatrix::build() {
#ifdef PARALLEL
    cudaMalloc((&this->mData), matrix_size * sizeof(float));
    cudaCheckError("Failed to build weight matrix!");
#else
    mData = (float*)malloc(matrix_size * sizeof(float));
    if (mData == NULL)
        throw "Failed to build weight matrix!";
#endif
    this->randomize(this->max_weight);
}

void WeightMatrix::randomize(float max_weight) {
#ifdef PARALLEL
    float* temp_matrix = (float*)malloc(matrix_size * sizeof(float));
    if (temp_matrix == NULL)
        throw "Failed to allocate temporary matrix on host!";

#endif
    for (int index = 0 ; index < matrix_size ; ++index) {
#ifdef PARALLEL
        temp_matrix[index] = fRand(0, max_weight);
#else
        this->mData[index] = fRand(0, max_weight);
#endif
    }

#ifdef PARALLEL
    cudaMemcpy(this->mData, temp_matrix, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to randomize weight matrix!");
    free(temp_matrix);
#else
#endif
}
