#include <algorithm>

#include "state/weight_matrix.h"
#include "model/layer.h"
#include "model/connection.h"
#include "util/error_manager.h"

/* Sets all values in an array to the given val */
void set_weights(float* arr, int size, float val, float fraction) {
    fSet(arr, size, val, fraction);
}

/* Clears an array */
void clear_weights(float* arr, int size) {
    set_weights(arr, size, 0.0);
}

/* Randomizes an array */
void randomize_weights(float* arr, int size, float max, float fraction) {
    fRand(arr, size, 0, max, fraction);
}
void randomize_weights_gaussian(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    // If standard deviation is 0.0, just set the weights to the mean
    if (std_dev == 0.0) {
        set_weights(arr, size, mean, fraction);
    } else {
        std::normal_distribution<double> dist(mean, std_dev);

        if (fraction == 1.0) {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = std::min((double)max, std::max(0.0, dist(generator)));
        } else {
            std::uniform_real_distribution<double> f_dist(0.0, 1.0);
            for (int i = 0 ; i < size ; ++i)
                arr[i] = (f_dist(generator) < fraction)
                    ? std::min((double)max, std::max(0.0, dist(generator)))
                    : 0.0;
        }
    }
}
void randomize_weights_lognormal(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    // If standard deviation is 0.0, just set the weights to the mean
    if (std_dev == 0.0) {
        set_weights(arr, size, mean, fraction);
    } else {
        std::lognormal_distribution<double> dist(mean, std_dev);

        if (fraction == 1.0) {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = std::min((double)max, std::max(0.0, dist(generator)));
        } else {
            std::uniform_real_distribution<double> f_dist(0.0, 1.0);
            for (int i = 0 ; i < size ; ++i)
                arr[i] = (f_dist(generator) < fraction)
                    ? std::min((double)max, std::max(0.0, dist(generator)))
                    : 0.0;
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

BasePointer* WeightMatrix::get_pointer() {
    return &this->mData;
}

void set_delays(OutputType output_type, Connection *conn,
        float* delays, float velocity,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset) {
    if (output_type != BIT)
        ErrorManager::get_instance()->log_error(
            "Only BIT output connections can have variable delays!");
    int base_delay = conn->delay;

    switch(conn->type) {
        case(FULLY_CONNECTED):
        case(SUBSET): {
            int from_row_start, from_col_start, to_row_start, to_col_start;
            int from_row_end, from_col_end, to_row_end, to_col_end;

            if (conn->type == FULLY_CONNECTED) {
                from_row_start = from_col_start = to_row_start = to_col_start = 0;
                from_row_end = conn->from_layer->rows;
                from_col_end = conn->from_layer->columns;
                to_row_end = conn->to_layer->rows;
                to_col_end = conn->to_layer->columns;
            } else if (conn->type == SUBSET) {
                auto sc = conn->get_config()->get_subset_config();
                from_row_start = sc->from_row_start;
                from_col_start = sc->from_col_start;
                to_row_start = sc->to_row_start;
                to_col_start = sc->to_col_start;
                from_row_end = sc->from_row_end;
                from_col_end = sc->from_col_end;
                to_row_end = sc->to_row_end;
                to_col_end = sc->to_col_end;
            }

            int from_columns = conn->from_layer->columns;
            int weight_index = 0;
            for (int f_row = from_row_start; f_row < from_row_end; ++f_row) {
                float f_y = f_row * from_spacing;

                for (int f_col = from_col_start; f_col < from_col_end; ++f_col) {
                    float f_x = f_col * from_spacing;

                    for (int t_row = to_row_start; t_row < to_row_end; ++t_row) {
                        float t_y = t_row * to_spacing + y_offset;

                        for (int t_col = to_col_start; t_col < to_col_end; ++t_col) {
                            float t_x = t_col * to_spacing + x_offset;

                            float distance = pow(
                                pow(t_x - f_x, 2) + pow(t_y - f_y, 2),
                                0.5);
                            int delay = base_delay + (distance / velocity);
                            if (delay > 31)
                                ErrorManager::get_instance()->log_error(
                                    "Unmyelinated axons cannot have delays "
                                    "greater than 31!");
                            delays[weight_index] = delay;
                            ++weight_index;
                        }
                    }
                }
            }
            break;
        }
        case(ONE_TO_ONE):
            for (int i = 0 ; i < conn->get_num_weights() ; ++i)
                delays[i] = base_delay;
            break;
        case(CONVERGENT): {
            auto ac = conn->get_config()->get_arborized_config();
            int to_size = conn->to_layer->size;
            int field_size = ac->row_field_size * ac->column_field_size;

            if (ac->row_stride != ac->column_stride
                or (int(to_spacing / from_spacing) != ac->row_stride))
                ErrorManager::get_instance()->log_error(
                    "Spacing and strides must match up for convergent connection!");

            for (int f_row = 0; f_row < ac->row_field_size; ++f_row) {
                float f_y = (f_row + ac->row_offset) * to_spacing;

                for (int f_col = 0; f_col < ac->column_field_size; ++f_col) {
                    float f_x = (f_col + ac->column_offset) * to_spacing;

                    float distance = pow(
                        pow(f_x, 2) + pow(f_y, 2),
                        0.5);
                    int delay = base_delay + (distance / velocity);
                    if (delay > 31)
                        ErrorManager::get_instance()->log_error(
                            "Unmyelinated axons cannot have delays "
                            "greater than 31!");

                    int f_index = (f_row * ac->column_field_size) + f_col;

                    for (int i = 0 ; i < to_size ; ++i) {
#ifdef __CUDACC__
                        delays[(f_index * to_size) + i] = delay;
#else
                        delays[(i * field_size) + f_index] = delay;
#endif
                    }
                }
            }
            break;
        }
        case(DIVERGENT): {
            auto ac = conn->get_config()->get_arborized_config();
            int num_weights = conn->get_num_weights();
            int to_rows = conn->to_layer->rows;
            int to_columns = conn->to_layer->columns;
            int to_size = conn->to_layer->size;
            int from_rows = conn->from_layer->rows;
            int from_columns = conn->from_layer->columns;

            if (ac->row_stride != ac->column_stride
                or (int(from_spacing / to_spacing) != ac->row_stride))
                ErrorManager::get_instance()->log_error(
                    "Spacing and strides must match up for divergent connection!");


            int row_field_size = ac->row_field_size;
            int column_field_size = ac->column_field_size;
            int row_stride = ac->row_stride;
            int column_stride = ac->column_stride;
            int row_offset = ac->row_offset;
            int column_offset = ac->column_offset;
            int kernel_size = row_field_size * column_field_size;

            for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
                for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
                    int to_index = d_row*to_columns + d_col;

                    /* Determine range of source neurons for divergent kernel */
                    int start_s_row = (d_row - row_offset - row_field_size + row_stride) / row_stride;
                    int start_s_col = (d_col - column_offset - column_field_size + column_stride) / column_stride;
                    int end_s_row = (d_row - row_offset) / row_stride;
                    int end_s_col = (d_col - column_offset) / column_stride;

                    // SERIAL
                    int weight_offset = to_index * num_weights / to_size;
                    // PARALLEL
                    int kernel_row_size = num_weights / to_size;

                    /* Iterate over relevant source neurons... */
                    int k_index = 0;
                    for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                        for (int s_col = start_s_col ; s_col <= end_s_col ; (++s_col, ++k_index)) {
                            /* Avoid making connections with non-existent neurons! */
                            if (s_row < 0 or s_row >= from_rows
                                or s_col < 0 or s_col >= from_columns)
                                continue;

                            int from_index = (s_row * from_columns) + s_col;

                            float d_x = abs(
                                ((d_row + ac->row_offset) * to_spacing)
                                - (s_row * from_spacing));
                            float d_y = abs(
                                ((d_col + ac->column_offset) * to_spacing)
                                - (s_col * from_spacing));

                            float distance = pow(
                                pow(d_x, 2) + pow(d_y, 2),
                                0.5);
                            int delay = base_delay + (distance / velocity);
                            if (delay > 31)
                                ErrorManager::get_instance()->log_error(
                                    "Unmyelinated axons cannot have delays "
                                    "greater than 31!");
#ifdef __CUDACC__
                            int weight_index = to_index + (k_index * kernel_row_size);
#else
                            int weight_index = weight_offset + k_index;
#endif
                            delays[weight_index] = delay;
                        }
                    }
                }
            }
            break;
        }
    }
}
