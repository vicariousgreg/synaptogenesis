#ifndef matrix_h
#define matrix_h

#include <cstdlib>
#include <stdio.h>

#ifdef parallel
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

class Matrix
{
    public:
        Matrix(int rows, int cols) :
                mRows(rows),
                mCols(cols) {
#ifdef parallel
            cudaMalloc(((void**)&this->mData), rows * cols * sizeof(double));
#else
            mData = (double*)malloc(rows * cols * sizeof(double));
#endif
        }

        void randomize(bool self_connected, double max_weight);

        double& operator()(int i, int j) {
            return this->mData[i * mCols + j];
        }

        double operator()(int i, int j) const {
            return this->mData[i * mCols + j];
        }

        double* mData;

    private:
        int mRows;
        int mCols;
};

#endif
