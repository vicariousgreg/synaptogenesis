#include <cstdlib>

#include "driver.h"
#include "model.h"
#include "tools.h"
#include "parallel.h"

/* Randomizes matrices */
void randomize_matrices(Model* model, float** entry_points, int depth) {
    for (int i = 0 ; i < model->num_connections ; ++i) {
        Connection &conn = model->connections[i];
        // Skip shared connections
        if (conn.parent != -1) continue;
        float* mData = entry_points[i];

        // Multiply by depth if plastic
        int matrix_size = conn.num_weights;
        if (conn.plastic) matrix_size *= depth;

#ifdef PARALLEL
        float* temp_matrix = (float*)calloc(matrix_size, sizeof(float));
        if (temp_matrix == NULL)
            throw "Failed to allocate temporary matrix on host for randomization!";

#endif
        // Randomize the first layer of the matrix (weights)
        // Further layers are initialized to zero.
        for (int index = 0 ; index < conn.num_weights ; ++index) {
#ifdef PARALLEL
            temp_matrix[index] = fRand(0, conn.max_weight);
#else
            mData[index] = fRand(0, conn.max_weight);
#endif
        }

        // Set up further layers if necessary (initialize to zero)
        for (int index = conn.num_weights ; index < matrix_size ; ++index) {
#ifdef PARALLEL
            temp_matrix[index] = 0.0;
#else
            mData[index] = 0.0;
#endif
        }

#ifdef PARALLEL
        cudaMemcpy(mData, temp_matrix,
            matrix_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckError("Failed to randomize weight matrix!");
        free(temp_matrix);
#endif
    }
}

float** build_weight_matrices(Model* model, int depth) {
    // Allocate one big glob for weights
    // Skip shared weights because they don't take up extra space
    int total_size = 0;
    for (int i = 0 ; i < model->num_connections ; ++i) {
        int matrix_size = 0;
        if (model->connections[i].parent == -1)
            total_size += model->connections[i].num_weights;

        // If plastic, multiply by depth to make room.
        if (model->connections[i].plastic)
            matrix_size *= depth;
    }

    float* matrix_datas;
#ifdef PARALLEL
    cudaMalloc((&matrix_datas), total_size * sizeof(float));
    cudaCheckError("Failed to allocate space for weight matrices!");
#else
    matrix_datas = (float*)malloc(total_size * sizeof(float));
    if (matrix_datas == NULL)
        throw "Failed to allocate space for weight matrices!";
#endif

    // Allocate double pointer for indexing purposes
    float** entry_points = 
        (float**)malloc(model->num_connections * sizeof(float*));
    float* curr_point = matrix_datas;
    for (int i = 0 ; i < model->num_connections ; ++i) {
        Connection &conn = model->connections[i];
        if (conn.parent == -1) {
            entry_points[i] = curr_point;
            // TODO: factor in weight depth for LTP, LTD, etc
            curr_point += conn.num_weights;
        } else {
            entry_points[i] = entry_points[conn.parent];
        }
    }
    randomize_matrices(model, entry_points, depth);
    return entry_points;
}
