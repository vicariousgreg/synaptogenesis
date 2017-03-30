#include <sstream>

#include "state/weight_matrix.h"
#include "model/layer.h"
#include "util/error_manager.h"

/* Sets all values in an array to the given val */
void set_weights(float* arr, int size, float val) {
    for (int i = 0 ; i < size ; ++i) arr[i] = val;
}

/* Clears an array */
void clear_weights(float* arr, int size) {
    set_weights(arr, size, 0.0);
}

/* Randomizes an array */
void randomize_weights(float* arr, int size, float max, float fraction) {
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = fRand(0, max);
    } else {
        for (int i = 0 ; i < size ; ++i)
            if (fRand(0,1) < fraction)
                arr[i] = fRand(0, max);
            else
                arr[i] = 0.0;
    }
}

/* Transfers the values from one array to another */
void transfer_weights(float* from, float* to, int size) {
    for (int i = 0 ; i < size ; ++i) to[i] = from[i];
}

WeightMatrix::WeightMatrix(Connection* conn, int matrix_depth,
        DeviceID device_id) : connection(conn), device_id(device_id) {
    int num_weights = conn->get_num_weights();
    matrix_size = num_weights;
    // Multiply by depth if plastic
    if (conn->plastic) matrix_size *= matrix_depth;

    // Allocate matrix on host
    // If parallel, it will be copied below
    mData = Pointer<float>(matrix_size);
    if (mData.get() == nullptr)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space for weight matrices on host!");

    // If parameter is specified, interpret it for initialization
    // Otherwise, perform randomization
    conn->get_weight_config()->initialize(mData, conn,
        ResourceManager::get_instance()->is_host(device_id));
}

WeightMatrix::~WeightMatrix() {
    this->mData.free();
}

void WeightMatrix::schedule_transfer() {
    this->mData.schedule_transfer(device_id);
}
