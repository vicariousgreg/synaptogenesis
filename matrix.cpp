#include "matrix.h"
#include "tools.h"

void Matrix::randomize(bool self_connected, double max_weight) {
#ifdef parallel
    int matrix_size = this->mRows * this->mCols;
    double* temp_matrix = (double*)malloc(matrix_size * sizeof(double));
#endif
    for (int row = 0 ; row < this->mRows ; ++row) {
        for (int col = 0 ; col < this->mCols ; ++col) {
            if (self_connected || (row != col))
#ifdef parallel
                temp_matrix[row * mCols + col] = fRand(0, max_weight);
#else
                this->mData[row * mCols + col] = fRand(0, max_weight);
#endif
        }
    }

#ifdef parallel
    cudaMemcpy(this->mData, temp_matrix, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    free(temp_matrix);
#endif
}
