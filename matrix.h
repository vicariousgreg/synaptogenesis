#ifndef matrix_h
#define matrix_h

#include <cstdlib>

class Matrix
{
    public:
        Matrix(int rows, int cols) :
                mRows(rows),
                mCols(cols) {
            mData = (double*)malloc(rows * cols * sizeof(double));
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
