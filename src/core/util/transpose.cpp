#include "util/transpose.h"
#include "util/pointer.h"

#include <vector>

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


dim3 calc_transpose_threads(int original_rows, int original_columns) {
    return dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS, 1);
}

dim3 calc_transpose_blocks(int original_rows, int original_columns) {
    return dim3(
        (original_columns/TRANSPOSE_TILE_DIM)
            + (original_columns % TRANSPOSE_TILE_DIM > 0),
        (original_rows/TRANSPOSE_TILE_DIM)
            + (original_rows % TRANSPOSE_TILE_DIM > 0), 1);
}

#ifdef __CUDACC__

template GLOBAL void transpose_matrix_parallel<float>(
	const Pointer<float> idata, Pointer<float> odata,
	const int original_rows, const int original_columns);
template GLOBAL void transpose_matrix_parallel<int>(
	const Pointer<int> idata, Pointer<int> odata,
	const int original_rows, const int original_columns);

template<class T>
GLOBAL void transpose_matrix_parallel(
        const Pointer<T> idata, Pointer<T> odata,
        const int original_rows, const int original_columns) {
    __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM+1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    T* in = idata.get();
    if (x < original_columns)
        for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
            if (y+j < original_rows)
                tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*original_columns + x];

    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    T* out = odata.get();
    if (x < original_rows)
        for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
            if (y + j < original_columns)
                out[(y+j)*original_rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

#endif
