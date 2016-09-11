#pragma once
#include <vector>
#include <cstdlib>
#include <iostream>

class Matrix
{
    public:
        Matrix(int rows, int cols)
        : mRows(rows),
          mCols(cols),
          mData(rows * cols) {}

        void randomize() {
            for (int row = 0 ; row < this->mRows ; ++row) {
                for (int col = 0 ; col < this->mCols ; ++col) {
                    this->mData[row * mCols + col] = (double)rand() / RAND_MAX;
                    //std::cout << this->mData[row * mCols + col];
                }
            }
        }

        double& operator()(int i, int j)
        {
            return this->mData[i * mCols + j];
        }

        double operator()(int i, int j) const
        {
            return this->mData[i * mCols + j];
        }

    private:
        int mRows;
        int mCols;
        std::vector<double> mData;

};
