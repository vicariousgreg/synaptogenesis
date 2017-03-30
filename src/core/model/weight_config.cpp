#include <sstream>

#include "model/weight_config.h"
#include "model/layer.h"
#include "model/connection.h"
#include "state/weight_matrix.h"
#include "util/error_manager.h"

void FlatWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) const {
    int num_weights = conn->get_num_weights();
    set_weights(target_matrix, num_weights, weight);
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
