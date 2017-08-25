#include <sstream>

#include "model/weight_config.h"
#include "model/layer.h"
#include "model/connection.h"
#include "state/weight_matrix.h"
#include "util/error_manager.h"

void WeightConfig::flat_config(float* target_matrix,
        Connection* conn, bool is_host) {
    float weight = std::stof(this->get_property("weight", "1.0"));
    float fraction = std::stof(this->get_property("fraction", "1.0"));

    set_weights(target_matrix, conn->get_num_weights(), weight, fraction);
}

void WeightConfig::random_config(float* target_matrix,
        Connection* conn, bool is_host) {
    float max_weight = std::stof(this->get_property("max weight", "1.0"));
    float fraction = std::stof(this->get_property("fraction", "1.0"));

    randomize_weights(target_matrix, conn->get_num_weights(),
        max_weight, fraction);
}

void WeightConfig::gaussian_config(float* target_matrix,
        Connection* conn, bool is_host) {
    float mean = std::stof(this->get_property("mean", "1.0"));
    float std_dev = std::stof(this->get_property("std dev", "0.3"));
    float fraction = std::stof(this->get_property("fraction", "1.0"));

    if (std_dev < 0)
        ErrorManager::get_instance()->log_error(
            "Error in weight config for " + conn->str() + ":\n"
            "  Gaussian weight config std_dev must be positive!");

    randomize_weights_gaussian(target_matrix, conn->get_num_weights(),
        mean, std_dev, conn->max_weight, fraction);
}

void WeightConfig::log_normal_config(float* target_matrix,
        Connection* conn, bool is_host) {
    float mean = std::stof(this->get_property("mean", "1.0"));
    float std_dev = std::stof(this->get_property("std dev", "0.3"));
    float fraction = std::stof(this->get_property("fraction", "1.0"));

    if (std_dev < 0)
        ErrorManager::get_instance()->log_error(
            "Error in weight config for " + conn->str() + ":\n"
            "  Log normal weight config std_dev must be positive!");

    randomize_weights_lognormal(target_matrix, conn->get_num_weights(),
        mean, std_dev, conn->max_weight, fraction);
}

void WeightConfig::surround_config(float* target_matrix,
        Connection* conn, bool is_host) {
    switch (conn->type) {
        case CONVERGENT:
        case CONVOLUTIONAL:
        case DIVERGENT:
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Error in weight config for " + conn->str() + ":\n"
                "  SurroundWeightConfig can only be used "
                "on arborized connections!");
    }

    int rows = std::stoi(this->get_property("rows", "0"));
    int cols = std::stoi(this->get_property("columns", "0"));
    int size_param = std::stoi(this->get_property("size", "-1"));

    if (size_param >= 0)
        rows = cols = size_param;
    if (rows < 0 or cols < 0)
        ErrorManager::get_instance()->log_error(
            "Error in weight config for " + conn->str() + ":\n"
            "  Surround weight config rows/cols must be positive!");

    // Initialize with child config
    auto child_config = this->get_child();
    if (child_config == nullptr)
        ErrorManager::get_instance()->log_error(
            "Error in weight config for " + conn->str() + ":\n"
            "  Missing child weight config for surround weight config!");
    child_config->initialize(target_matrix, conn, is_host);

    // Carve out center
    int row_field_size =
        conn->get_config()->get_arborized_config()->column_field_size;
    int col_field_size =
        conn->get_config()->get_arborized_config()->column_field_size;
    int kernel_size =
        conn->get_config()->get_arborized_config()->get_total_field_size();

    int row_offset = (row_field_size - rows) / 2;
    int col_offset = (col_field_size - cols) / 2;

    // Divergent connections are unique in that there is a kernel per source
    //   neuron.  All other connection types organize based on the
    //   destination layer.  Convolutional connections have only one kernel.
    int size;
    switch (conn->type) {
        case DIVERGENT: size = conn->from_layer->size; break;
        case CONVERGENT: size = conn->to_layer->size; break;
        case CONVOLUTIONAL: size = 1; break;
    }

    if (is_host) {
        for (int index = 0 ; index < size ; ++index) {
            int weight_offset = (conn->convolutional)
                ? 0 : (index * kernel_size);

            for (int k_row = row_offset ; k_row < row_offset + rows ; ++k_row) {
                for (int k_col = col_offset ;
                        k_col < col_offset + cols ; ++k_col) {
                    int weight_index = weight_offset +
                        (k_row * col_field_size) + k_col;
                    target_matrix[weight_index] = 0.0;
                }
            }
        }
    } else {
        for (int index = 0 ; index < size ; ++index) {
            int weight_col = (conn->convolutional) ? 0 : index;
            int kernel_row_size = (conn->convolutional) ? 1 : size;

            for (int k_row = row_offset ; k_row < row_offset + rows ; ++k_row) {
                for (int k_col = col_offset ;
                        k_col < col_offset + cols ; ++k_col) {
                    int weight_index = weight_col +
                        ((k_row * col_field_size) + k_col) * kernel_row_size;
                    target_matrix[weight_index] = 0.0;
                }
            }
        }
    }
}

void WeightConfig::specified_config(float* target_matrix,
        Connection* conn, bool is_host) {
    std::string weight_string = this->get_property("weight string", "");
    if (weight_string == "")
        ErrorManager::get_instance()->log_error(
            "Error in weight config for " + conn->str() + ":\n"
            "  Missing weight string for specified weight config!");

    std::stringstream stream(weight_string);
    int num_weights = conn->get_num_weights();

    int rows = conn->to_layer->size;
    int cols = conn->from_layer->size;
    int row_field_size =
        conn->get_config()->get_arborized_config()->row_field_size;
    int column_field_size =
        conn->get_config()->get_arborized_config()->column_field_size;
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
                    "Error in weight config for " + conn->str() + ":\n"
                    "  Insufficient number of weights specified!");
            else stream >> value;

            // If parallel, transpose the input (rows <-> cols)
            // Parallel convergent matrices are laid out such that each
            //   kernel is in a column
            if (is_host) target_matrix[row * cols + col] = value;
            else         target_matrix[col * rows + row] = value;
        }
    }
}

void WeightConfig::initialize(float* target_matrix,
        Connection* conn, bool is_host) {
    if (this->has_property("fraction")) {
        float fraction = std::stof(this->get_property("fraction", "1.0"));
        if (fraction < 0 or fraction > 1.0)
            ErrorManager::get_instance()->log_error(
                "Error in weight config for " + conn->str() + ":\n"
                "  Weight config fraction must be between 0 and 1!");
    }

    auto type = this->get_property("type");

    if (type == "flat")
        flat_config(target_matrix, conn, is_host);
    else if (type == "random")
        random_config(target_matrix, conn, is_host);
    else if (type == "gaussian")
        gaussian_config(target_matrix, conn, is_host);
    else if (type == "log normal")
        log_normal_config(target_matrix, conn, is_host);
    else if (type == "surround")
        surround_config(target_matrix, conn, is_host);
    else if (type == "specified")
        specified_config(target_matrix, conn, is_host);
    else
        ErrorManager::get_instance()->log_error(
            "Error in weight config for " + conn->str() + ":\n"
            "  Unrecognized weight config type: " + type);

    if (this->get_property("diagonal", "true") != "true") {
        switch(conn->type) {
            case FULLY_CONNECTED:
                clear_diagonal(target_matrix,
                    conn->from_layer->size, conn->to_layer->size);
                break;
            case SUBSET: {
                auto subset_config = conn->get_config()->get_subset_config();
                clear_diagonal(target_matrix,
                    subset_config->from_size, subset_config->to_size);
                break;
            }
            case CONVERGENT:
                break;
        }
    }
}
