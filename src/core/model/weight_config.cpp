#include <sstream>

#include "model/weight_config.h"
#include "model/layer.h"
#include "model/connection.h"
#include "state/weight_matrix.h"
#include "util/error_manager.h"

FlatWeightConfig::FlatWeightConfig(float weight, float fraction)
        : weight(weight), fraction(fraction) {
    if (fraction < 0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "RandomWeightConfig fraction must be between 0 and 1!");
}

void FlatWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) const {
    int num_weights = conn->get_num_weights();
    set_weights(target_matrix, num_weights, weight, fraction);
}

RandomWeightConfig::RandomWeightConfig(float max_weight, float fraction)
        : max_weight(max_weight), fraction(fraction) {
    if (fraction < 0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "RandomWeightConfig fraction must be between 0 and 1!");
}

void RandomWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) const {
    int num_weights = conn->get_num_weights();
    randomize_weights(target_matrix, num_weights, max_weight, fraction);
}

SurroundWeightConfig::SurroundWeightConfig(
        int rows, int cols, WeightConfig *base_config)
        : rows(rows), cols(cols), base_config(base_config) {
    if (rows < 0 or cols < 0)
        ErrorManager::get_instance()->log_error(
            "SurroundWeightConfig rows/cols must be positive!");
}
SurroundWeightConfig::SurroundWeightConfig(
        int size, WeightConfig *base_config)
        : SurroundWeightConfig(size, size, base_config) { }

SurroundWeightConfig::~SurroundWeightConfig() {
    delete base_config;
}

void SurroundWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) const {
    switch (conn->type) {
        case(CONVERGENT):
        case(DIVERGENT):
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "SurroundWeightConfig can only be used on arborized connections!");
    }

    // Initialize with base config
    base_config->initialize(target_matrix, conn, is_host);

    // Carve out center
    int row_field_size = conn->get_column_field_size();
    int col_field_size = conn->get_column_field_size();
    int kernel_size = conn->get_total_field_size();

    int row_offset = (row_field_size - rows) / 2;
    int col_offset = (col_field_size - cols) / 2;

    if (is_host) {
        for (int to_index = 0 ; to_index < conn->to_layer->size ; ++to_index) {
            int weight_offset = (conn->convolutional) ? 0 : (to_index * kernel_size);

            for (int k_row = row_offset ; k_row < row_offset + rows ; ++k_row) {
                for (int k_col = col_offset ; k_col < col_offset + cols ; ++k_col) {
                    int weight_index = weight_offset + (k_row * col_field_size) + k_col;
                    target_matrix[weight_index] = 0.0;
                }
            }
        }
    } else {
        for (int to_index = 0 ; to_index < conn->to_layer->size ; ++to_index) {
            int weight_col = (conn->convolutional) ? 0 : to_index;
            int kernel_row_size = (conn->convolutional) ? 1 : conn->to_layer->size;

            for (int k_row = row_offset ; k_row < row_offset + rows ; ++k_row) {
                for (int k_col = col_offset ; k_col < col_offset + cols ; ++k_col) {
                    int weight_index = weight_col +
                        ((k_row * col_field_size) + k_col) * kernel_row_size;
                    target_matrix[weight_index] = 0.0;
                }
            }
        }
    }
}

void SpecifiedWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) const {
    std::stringstream stream(weight_string);
    int num_weights = conn->get_num_weights();

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

    float value;
    for (int row = 0 ; row < rows ; ++row) {
        for (int col = 0 ; col < cols ; ++col) {
            if (row != rows-1 and col != cols-1 and stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Insufficient number of weights specified!");
            else stream >> value;

            // If parallel, transpose the input (rows <-> cols)
            // Parallel convergent matrices are laid out such that each
            //   kernel is in a column
            if (is_host) target_matrix[row * cols + col] = value;
            else         target_matrix[col * rows + row] = value;
        }
    }
}
