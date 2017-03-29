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
void randomize_weights(float* arr, int size, float max) {
    for (int i = 0 ; i < size ; ++i) arr[i] = fRand(0, max);
}

/* Transfers the values from one array to another */
void transfer_weights(float* from, float* to, int size) {
    for (int i = 0 ; i < size ; ++i) to[i] = from[i];
}

/* Initializes weights using connection parameters */
static void initialize(float* target_matrix, Connection* conn,
        DeviceID device_id) {
    std::stringstream stream(conn->get_init_params());
    int num_weights = conn->get_num_weights();

    // Extract first value
    float value;
    stream >> value;

    // Check if this weight matrix is staying on the host
    bool is_host = ResourceManager::get_instance()->is_host(device_id);

    // If there are no more values, set all weights to first value
    // Otherwise, read values and initialize
    if (stream.eof()) {
        set_weights(target_matrix, num_weights, value);
    } else {
        int rows = conn->to_layer->size;
        int cols = conn->from_layer->size;
        int row_field_size = conn->get_row_field_size();
        int column_field_size = conn->get_column_field_size();
        switch (conn->type) {
            case CONVOLUTIONAL:
                rows = 1;
                cols = row_field_size * column_field_size;
                break;
            case CONVERGENT:
                rows = conn->to_layer->size;
                cols = row_field_size * column_field_size;
                break;
            case DIVERGENT:
                rows = conn->from_layer->size;
                cols = row_field_size * column_field_size;
                break;
            case ONE_TO_ONE:
                rows = 1;
                break;
        }

        for (int row = 0 ; row < rows ; ++row) {
            for (int col = 0 ; col < cols ; ++col) {
                // If parallel, transpose the input (rows <-> cols)
                // Parallel convergent matrices are laid out such that each
                //   kernel is in a column
                if (is_host) target_matrix[row * cols + col] = value;
                else         target_matrix[col * rows + row] = value;

                if (row != rows-1 and col != cols-1 and stream.eof())
                    ErrorManager::get_instance()->log_error(
                        "Insufficient number of weights specified!");
                else stream >> value;
            }
        }
    }
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
    if (conn->get_init_params().size() == 0)
        randomize_weights(mData, num_weights, conn->max_weight);
    else
        initialize(mData, conn, device_id);
}

WeightMatrix::~WeightMatrix() {
    this->mData.free();
}

void WeightMatrix::schedule_transfer() {
    this->mData.schedule_transfer(device_id);
}
