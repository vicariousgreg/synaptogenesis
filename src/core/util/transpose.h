#ifndef transpose_h
#define transpose_h

#include "parallel.h"

/* Transposes a matrix in place */
template <typename T>
void transpose_matrix_in_place(T* data, int original_rows, int original_cols);


const int TRANSPOSE_TILE_DIM = 32;
const int TRANSPOSE_BLOCK_ROWS = 8;

dim3 calc_transpose_threads(int original_rows, int original_columns);
dim3 calc_transpose_blocks(int original_rows, int original_columns);

#ifdef __CUDACC__

template<class T>
class Pointer;

/* Transposes a matrix out of place on a device */
template<class T>
GLOBAL void transpose_matrix_parallel(
    const Pointer<T> idata, Pointer<T> odata,
	const int original_rows, const int original_columns);

#endif

#endif
