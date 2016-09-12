#ifndef matrix_h
#define matrix_h

#include <vector>
#include <cstdlib>

class Matrix
{
    public:
        Matrix(int rows, int cols) :
                mRows(rows),
                mCols(cols),
                mData(rows * cols) {}

        void randomize();

        double& operator()(int i, int j) {
            return this->mData[i * mCols + j];
        }

        double operator()(int i, int j) const {
            return this->mData[i * mCols + j];
        }

    private:
        int mRows;
        int mCols;
        std::vector<double> mData;
};

#endif
