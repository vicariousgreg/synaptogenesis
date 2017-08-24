#include <sstream>

#include "model/weight_config.h"
#include "model/layer.h"
#include "model/connection.h"
#include "state/weight_matrix.h"
#include "util/error_manager.h"

void WeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    if (not diagonal) {
        switch(conn->type) {
            case(FULLY_CONNECTED):
                clear_diagonal(target_matrix, conn->from_layer->size, conn->to_layer->size);
                break;
            case(SUBSET): {
                auto subset_config = conn->get_config()->get_subset_config();
                clear_diagonal(target_matrix, subset_config->from_size, subset_config->to_size);
                break;
            }
            case(CONVERGENT):
                break;
        }
    }
}

FlatWeightConfig::FlatWeightConfig(float weight, float fraction)
        : WeightConfig("flat"), weight(weight), fraction(fraction) {
    this->set_property("weight", std::to_string(weight));
    this->set_property("fraction", std::to_string(fraction));
    if (fraction < 0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "FlatWeightConfig fraction must be between 0 and 1!");
}

void FlatWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    int num_weights = conn->get_num_weights();
    set_weights(target_matrix, num_weights, weight, fraction);
    WeightConfig::initialize(target_matrix, conn, is_host);
}

RandomWeightConfig::RandomWeightConfig(float max_weight, float fraction)
        : WeightConfig("random"), max_weight(max_weight), fraction(fraction) {
    this->set_property("max weight", std::to_string(max_weight));
    this->set_property("fraction", std::to_string(fraction));

    if (fraction < 0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "RandomWeightConfig fraction must be between 0 and 1!");
}

void RandomWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    int num_weights = conn->get_num_weights();
    randomize_weights(target_matrix, num_weights, max_weight, fraction);
    WeightConfig::initialize(target_matrix, conn, is_host);
}

GaussianWeightConfig::GaussianWeightConfig(float mean, float std_dev, float fraction)
        : WeightConfig("gaussian"), mean(mean), std_dev(std_dev), fraction(fraction) {
    this->set_property("mean", std::to_string(mean));
    this->set_property("std dev", std::to_string(std_dev));
    this->set_property("fraction", std::to_string(fraction));

    if (fraction < 0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "GaussianWeightConfig fraction must be between 0 and 1!");
    if (mean < 0 or std_dev < 0)
        ErrorManager::get_instance()->log_error(
            "GaussianWeightConfig mean and std_dev must be positive!");
}

void GaussianWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    int num_weights = conn->get_num_weights();
    randomize_weights_gaussian(target_matrix, num_weights,
        mean, std_dev, conn->max_weight, fraction);
    WeightConfig::initialize(target_matrix, conn, is_host);
}

LogNormalWeightConfig::LogNormalWeightConfig(float mean, float std_dev, float fraction)
        : WeightConfig("log normal"), mean(mean), std_dev(std_dev), fraction(fraction) {
    this->set_property("mean", std::to_string(mean));
    this->set_property("std dev", std::to_string(std_dev));
    this->set_property("fraction", std::to_string(fraction));

    if (fraction < 0 or fraction > 1.0)
        ErrorManager::get_instance()->log_error(
            "LogNormalWeightConfig fraction must be between 0 and 1!");
    if (std_dev < 0)
        ErrorManager::get_instance()->log_error(
            "LogNormalWeightConfig std_dev must be positive!");
}

void LogNormalWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    int num_weights = conn->get_num_weights();
    randomize_weights_lognormal(target_matrix, num_weights,
        mean, std_dev, conn->max_weight, fraction);
    WeightConfig::initialize(target_matrix, conn, is_host);
}

SurroundWeightConfig::SurroundWeightConfig(
        int rows, int cols, WeightConfig *child_config)
        : WeightConfig("surround"), rows(rows), cols(cols), child_config(child_config) {
    this->set_property("rows", std::to_string(rows));
    this->set_property("columns", std::to_string(cols));

    if (rows < 0 or cols < 0)
        ErrorManager::get_instance()->log_error(
            "SurroundWeightConfig rows/cols must be positive!");
}
SurroundWeightConfig::SurroundWeightConfig(
        int size, WeightConfig *child_config)
        : SurroundWeightConfig(size, size, child_config) { }

SurroundWeightConfig::~SurroundWeightConfig() {
    delete child_config;
}

void SurroundWeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    switch (conn->type) {
        case(CONVERGENT):
        case(CONVOLUTIONAL):
        case(DIVERGENT):
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "SurroundWeightConfig can only be used on arborized connections!");
    }

    // Initialize with child config
    child_config->initialize(target_matrix, conn, is_host);

    // Carve out center
    int row_field_size = conn->get_config()->get_arborized_config()->column_field_size;
    int col_field_size = conn->get_config()->get_arborized_config()->column_field_size;
    int kernel_size = conn->get_config()->get_arborized_config()->get_total_field_size();

    int row_offset = (row_field_size - rows) / 2;
    int col_offset = (col_field_size - cols) / 2;

    // Divergent connections are unique in that there is a kernel per source
    //   neuron.  All other connection types organize based on the
    //   destination layer.  Convolutional connections have only one kernel.
    int size;
    switch (conn->type) {
        case(DIVERGENT): size = conn->from_layer->size; break;
        case(CONVERGENT): size = conn->to_layer->size; break;
        case(CONVOLUTIONAL): size = 1; break;
    }

    if (is_host) {
        for (int index = 0 ; index < size ; ++index) {
            int weight_offset = (conn->convolutional) ? 0 : (index * kernel_size);

            for (int k_row = row_offset ; k_row < row_offset + rows ; ++k_row) {
                for (int k_col = col_offset ; k_col < col_offset + cols ; ++k_col) {
                    int weight_index = weight_offset + (k_row * col_field_size) + k_col;
                    target_matrix[weight_index] = 0.0;
                }
            }
        }
    } else {
        for (int index = 0 ; index < size ; ++index) {
            int weight_col = (conn->convolutional) ? 0 : index;
            int kernel_row_size = (conn->convolutional) ? 1 : size;

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
        Connection* conn, bool is_host) {
    std::stringstream stream(weight_string);
    int num_weights = conn->get_num_weights();

    int rows = conn->to_layer->size;
    int cols = conn->from_layer->size;
    int row_field_size = conn->get_config()->get_arborized_config()->row_field_size;
    int column_field_size = conn->get_config()->get_arborized_config()->column_field_size;
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

    WeightConfig::initialize(target_matrix, conn, is_host);
}
