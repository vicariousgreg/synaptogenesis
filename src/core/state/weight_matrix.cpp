#include <algorithm>

#include "state/weight_matrix.h"
#include "model/layer.h"
#include "model/connection.h"
#include "util/error_manager.h"

/* Sets all values in an array to the given val */
void set_weights(float* arr, int size, float val, float fraction) {
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = val;
    } else {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = (fRand() < fraction) ? val : 0.0;
    }
}

/* Clears an array */
void clear_weights(float* arr, int size) {
    set_weights(arr, size, 0.0);
}

/* Randomizes an array */
void randomize_weights(float* arr, int size, float max, float fraction) {
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = fRand(max);
    } else {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = (fRand() < fraction) ? fRand(max) : 0.0;
    }
}
void randomize_weights_gaussian(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    // If standard deviation is 0.0, just set the weights to the mean
    if (std_dev == 0.0) {
        set_weights(arr, size, mean, fraction);
    } else {
        std::normal_distribution<double> distribution(mean, std_dev);

        if (fraction == 1.0) {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = std::min((double)max, std::max(0.0, distribution(generator)));
        } else {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = (fRand() < fraction) ? std::min((double)max, std::max(0.0, distribution(generator))) : 0.0;
        }
    }
}
void randomize_weights_lognormal(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    // If standard deviation is 0.0, just set the weights to the mean
    if (std_dev == 0.0) {
        set_weights(arr, size, mean, fraction);
    } else {
        std::lognormal_distribution<double> distribution(mean, std_dev);

        if (fraction == 1.0) {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = std::min((double)max, std::max(0.0, distribution(generator)));
        } else {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = (fRand() < fraction) ? std::min((double)max, std::max(0.0, distribution(generator))) : 0.0;
        }
    }
}

/* Transfers the values from one array to another */
void transfer_weights(float* from, float* to, int size) {
    for (int i = 0 ; i < size ; ++i) to[i] = from[i];
}

/* Clears the diagonal of a weight matrix */
void clear_diagonal(float *arr, int rows, int cols) {
    if (rows != cols)
        ErrorManager::get_instance()->log_error(
            "Attempted to clear diagonal of non-square weight matrix!");

    for (int i = 0 ; i < rows ; ++i)
        arr[i * rows + i] = 0.0;
}

WeightMatrix::WeightMatrix(Connection* conn, int matrix_depth,
        DeviceID device_id) : connection(conn), device_id(device_id) {
    int num_weights = conn->get_num_weights();
    matrix_size = num_weights * matrix_depth;

    // Allocate matrix on host
    // If parallel, it will be copied below
    mData = Pointer<float>(matrix_size);
    if (mData.get() == nullptr)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space for weight matrices on host!");

    // If parameter is specified, interpret it for initialization
    // Otherwise, perform randomization
    conn->get_config()->weight_config->initialize(mData, conn,
        ResourceManager::get_instance()->is_host(device_id));
}

WeightMatrix::~WeightMatrix() {
    this->mData.free();
}

void WeightMatrix::schedule_transfer() {
    this->mData.schedule_transfer(device_id);
}
