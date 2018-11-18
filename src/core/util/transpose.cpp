#include "util/transpose.h"
#include "util/resources/pointer.h"

const int TRANSPOSE_TILE_DIM = 32;
const int TRANSPOSE_BLOCK_ROWS = 8;

template void transpose_matrix_in_place<float>(
    float* data, int original_rows, int original_cols, DeviceID device_id);
template void transpose_matrix_in_place<int>(
    int* data, int original_rows, int original_cols, DeviceID device_id);

template void transpose_matrices_in_place<float>(std::vector<float*> data,
        int original_rows, int original_cols, DeviceID device_id);
template void transpose_matrices_in_place<int>(std::vector<int*> data,
    int original_rows, int original_cols, DeviceID device_id);

template void transpose_matrix_out_of_place<float>(
    float* data, float* dest,
    int original_rows, int original_cols, DeviceID device_id);
template void transpose_matrix_out_of_place<int>(
    int* data, int* dest,
    int original_rows, int original_cols, DeviceID device_id);

#ifdef __CUDACC__
template GLOBAL void transpose_matrix_parallel<float>(
	const Pointer<float> idata, Pointer<float> odata,
	const int original_rows, const int original_columns);
template GLOBAL void transpose_matrix_parallel<int>(
	const Pointer<int> idata, Pointer<int> odata,
	const int original_rows, const int original_columns);
#endif


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


template <typename T>
void transpose_matrix_in_place(T* data,
        int original_rows, int original_cols, DeviceID device_id) {
    auto res_man = ResourceManager::get_instance();
    int size = original_rows * original_cols;

    if (res_man->is_host(device_id)) {
        transpose_in_place_impl(data, data + size, original_cols);
    } else {
#ifdef __CUDACC__
        // Create temporary matrix
        Pointer<T> temp = Pointer<T>::device_pointer(
            device_id, size);

        auto stream = res_man->get_default_stream(device_id);

        Pointer<T> p = Pointer<T>(data, size, device_id, false);
        p.copy_to(&temp, stream);
        transpose_matrix_out_of_place(temp.get_unsafe(),
            data, original_rows, original_cols, device_id);

        temp.free();
#endif
    }
}

template <typename T>
void transpose_matrices_in_place(std::vector<T*> data,
        int original_rows, int original_cols, DeviceID device_id) {
    auto res_man = ResourceManager::get_instance();
    int size = original_rows * original_cols;

    if (res_man->is_host(device_id)) {
        for (auto ptr : data)
            transpose_in_place_impl(ptr, ptr + size, original_cols);
    } else {
#ifdef __CUDACC__
        // Create temporary matrix
        Pointer<T> temp = Pointer<T>::device_pointer(device_id, size);

        auto stream = res_man->get_default_stream(device_id);

        for (auto ptr : data) {
            Pointer<T> p = Pointer<T>(ptr, size, device_id, false);
            p.copy_to(temp, stream);
            transpose_matrix_out_of_place(temp.get_unsafe(),
                ptr, original_rows, original_cols, device_id);
        }

        device_synchronize();
        temp.free();
#endif
    }
}


template <typename T>
void transpose_matrix_out_of_place(T* data, T* dest,
        int original_rows, int original_cols, DeviceID device_id) {
    auto res_man = ResourceManager::get_instance();
    if (res_man->is_host(device_id)) {
        for (int row = 0 ; row < original_rows ; ++row)
            for (int col = 0 ; col < original_cols ; ++col)
                dest[col * original_rows + row]
                    = data[row * original_cols + col];
    } else {
#ifdef __CUDACC__
        dim3 dimGrid = calc_transpose_blocks(original_rows, original_cols);
        dim3 dimBlock = calc_transpose_threads(original_rows, original_cols);
        auto stream = res_man->get_default_stream(device_id);

        cudaSetDevice(device_id);
        transpose_matrix_parallel<T>
            <<<dimGrid, dimBlock, 0, stream->get_cuda_stream()>>>
            (Pointer<T>(data, original_rows*original_cols, device_id, false),
            Pointer<T>(dest, original_rows*original_cols, device_id, false),
            original_rows, original_cols);

        device_check_error("Failed to transpose weight matrix!");
#endif
    }
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
