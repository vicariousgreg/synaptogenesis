#include "matrix.h"
#include "tools.h"

void Matrix::randomize(bool self_connected, double max_weight) {
    for (int row = 0 ; row < this->mRows ; ++row) {
        for (int col = 0 ; col < this->mCols ; ++col) {
            if (self_connected)
                this->mData[row * mCols + col] = fRand(0, max_weight);
        }
    }
}
