#include "matrix.h"
#include "tools.h"

void Matrix::randomize() {
    for (int row = 0 ; row < this->mRows ; ++row) {
        for (int col = 0 ; col < this->mCols ; ++col) {
            this->mData[row * mCols + col] = fRand(0, 1);
        }
    }
}
