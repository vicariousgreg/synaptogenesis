#include <cstdlib>
#include <cstdio>

#include "weight_matrix.h"
#include "tools.h"
#include "parallel.h"
#include "layer.h"
#include "constants.h"

WeightMatrix::WeightMatrix (Layer &from_layer, Layer &to_layer, bool plastic,
                int delay, float max_weight, MatrixType type, OPCODE opcode) :
            from_layer(from_layer),
            to_layer(to_layer),
            plastic(plastic),
            delay(delay),
            max_weight(max_weight),
            opcode(opcode),
            type(type),
            parent(NULL) {
    if (delay > (32 * HISTORY_SIZE - 1))
        throw "Cannot implement connection delay longer than history!";

    if (type == FULLY_CONNECTED) {
        this->matrix_size = from_layer.size * to_layer.size;
    } else if (type == ONE_TO_ONE) {
        if (from_layer.size != to_layer.size) {
            throw "Cannot connect differently sized layers one-to-one!";
        } else {
            this->matrix_size = from_layer.size;
        }
    }
}

WeightMatrix::WeightMatrix(Layer &from_layer, Layer &to_layer, WeightMatrix* parent) :
            from_layer(from_layer),
            to_layer(to_layer),
            parent(parent) {
    if (from_layer.size != parent->from_layer.size or
            to_layer.size != parent->to_layer.size) {
        throw "Cannot share weights between connections of different sizes!";
    }

    if (type == FULLY_CONNECTED) {
        this->matrix_size = from_layer.size * to_layer.size;
    } else if (type == ONE_TO_ONE) {
        this->matrix_size = from_layer.size;
    }
}

void WeightMatrix::build() {
    if (parent != NULL) {
        this->mData = parent->mData;
        this->plastic = parent->plastic;
        this->delay = parent->delay;
        this->max_weight = parent->max_weight;
        this->opcode = parent->opcode;
        this->type = parent->type;
    } else {
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
}

void WeightMatrix::randomize(float max_weight) {
    if (parent != NULL)
        throw "Cannot randomized weights via shared connection!";

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
