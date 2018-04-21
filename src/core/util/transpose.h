#ifndef transpose_h
#define transpose_h

#include <vector>

#include "parallel.h"

/* Transposes a matrix in place
 * Uses a temporary matrix for device transposes */
template <typename T>
void transpose_matrix_in_place(T* data,
    int original_rows, int original_cols, DeviceID device_id);

/* Transposes several matrices in place
 * Uses a temporary matrix for device transposes */
template <typename T>
void transpose_matrices_in_place(std::vector<T*> data,
    int original_rows, int original_cols, DeviceID device_id);

/* Transposes a matrix out of place (to another matrix) */
template <typename T>
void transpose_matrix_out_of_place(T* data, T* dest,
    int original_rows, int original_cols, DeviceID device_id);


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
