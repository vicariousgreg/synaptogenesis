#include "util/tools.h"

/* Static float function */
float fSet(float* arr, int size, float val, float fraction) {
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i) arr[i] = val;
    } else {
        auto dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (dist(generator) < fraction) arr[i] = val;
    }
}

void fRand(float* arr, int size, float fMin, float fMax, float fraction) {
    auto dist = std::uniform_real_distribution<float>(fMin, fMax);
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i) arr[i] = dist(generator);
    } else {
        auto f_dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (f_dist(generator) < fraction)
                arr[i] = dist(generator);
    }
}

void iRand(int* arr, int size, int iMin, int iMax, float fraction) {
    auto dist = std::uniform_int_distribution<int>(iMin,iMax);
    if (fraction == 1.0)
        for (int i = 0 ; i < size ; ++i) arr[i] = dist(generator);
    else {
        auto f_dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (f_dist(generator) < fraction)
                arr[i] = dist(generator);
    }
}

/******************************************************************************/
/**************************** MATRIX TRANSPOSITION ****************************/
/******************************************************************************/

/* Adapted from StackOverflow implementation of "Following the cycles" in-place
 *  transpose algorithm by Christian Ammer:
 * https://stackoverflow.com/questions/9227747/
 *     in-place-transposition-of-a-matrix
 */

template<class RandomIterator>
static void transpose_in_place_impl(RandomIterator first, RandomIterator last,
        long desired_rows) {
    const long mn1 = (last - first - 1);
    const long n   = (last - first) / desired_rows;
    std::vector<bool> visited(last - first);
    RandomIterator cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first]) continue;
        long a = cycle - first;
        do {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

template void transpose_matrix_in_place<float>(
    float* data, int original_rows, int original_cols);
template void transpose_matrix_in_place<int>(
    int* data, int original_rows, int original_cols);

template <typename T>
void transpose_matrix_in_place(T* data, int original_rows, int original_cols) {
    transpose_in_place_impl(data,
        data + (original_rows * original_cols), original_cols);
}
