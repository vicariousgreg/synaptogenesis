#include "io/impl/periodic_input_module.h"
#include "util/tools.h"

#include <sstream>
#include <iostream>
#include <cmath>

REGISTER_MODULE(BasicPeriodicInputModule, "periodic_input");
REGISTER_MODULE(OneHotRandomInputModule, "one_hot_random_input");
REGISTER_MODULE(OneHotCyclicInputModule, "one_hot_cyclic_input");
REGISTER_MODULE(GaussianRandomInputModule, "gaussian_random_input");

PeriodicInputModule::PeriodicInputModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config), dirty(true) {
    enforce_equal_layer_sizes("periodic_input");
    set_io_type(INPUT);

    this->value = config->get_float("value", 1.0);
    this->min_value = config->get_float("min", 0.0);
    this->max_value = config->get_float("max", 1.0);
    this->fraction = config->get_float("fraction", 1.0);
    this->random = config->get_bool("random", false);

    if (this->min_value > this->max_value)
        LOG_ERROR(
            "Invalid min/max value for periodic input generator!");
    if (this->fraction <= 0.0 or this->fraction > 1.0)
        LOG_ERROR(
            "Invalid fraction for periodic input generator!");

    this->values = Pointer<float>(layers.at(0)->size, 0.0);
}

PeriodicInputModule::~PeriodicInputModule() {
    this->values.free();
}

void PeriodicInputModule::feed_input_impl(Buffer *buffer) {
    if (dirty) {
        for (auto layer : layers)
            this->values.copy_to(buffer->get_input(layer));
        dirty = false;
    }
}

void PeriodicInputModule::cycle_impl() {
    if (curr_iteration % rate == 0) {
        dirty = true;
        this->update();
    }

    if (verbose) {
        std::cout << "============================ SHUFFLE\n";
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            std::cout << this->values[nid] << " ";
        std::cout << std::endl;
    }
}

void BasicPeriodicInputModule::update() {
    if (random)
        fRand(values, values.get_size(), min_value, max_value, fraction);
    else
        fSet(values, values.get_size(), value, fraction);
}

void OneHotRandomInputModule::update() {
    // Randomly selects one input to activate
    int random_index = iRand(0, values.get_size()-1);

    if (random)
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == random_index)
                ? fRand(min_value, max_value) : 0;
    else
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == random_index) ? value : 0;
}

void OneHotCyclicInputModule::update() {
    int index = curr_iteration % values.get_size();

    if (random)
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == index)
                ? fRand(min_value, max_value) : 0;
    else
        for (int nid = 0 ; nid < values.get_size(); ++nid)
            values[nid] =
                (nid == index) ? value : 0;
}

GaussianRandomInputModule::GaussianRandomInputModule(LayerList layers,
        ModuleConfig *config) : PeriodicInputModule(layers, config) {
    // Module::build_module checks for this, but to be safe...
    if (layers.size() == 0)
        LOG_ERROR("Attempted to initialize GaussianRandomInputModule"
            " with empty LayerList!");

    this->rows = layers.at(0)->rows;
    this->columns = layers.at(0)->columns;
    float std_dev = config->get_float("std dev", 1.0);
    bool normalize = config->get_bool("normalize", true);
    this->num_peaks = config->get_int("peaks", 1);

    /* Precompute gaussian values
     * If random, use 1.0 as peak
     * Allocate space to accommodate peaks at the corners */
    this->gauss_rows = rows * 2 - 1;
    this->gauss_columns = columns * 2 - 1;
    int row_center = rows - 1;
    int column_center = columns - 1;
    this->gaussians = Pointer<float>(gauss_rows * gauss_columns, 0.0);

    /* Gaussian algorithm adapted from:
     * https://stackoverflow.com/questions/10847007/
     *     using-the-gaussian-probability-density-function-in-c */
    static const float inv_sqrt_2pi = 0.3989422804014327;

    /* If random, peak will be multiplied online
     * If not normalizing, set the coefficient */
    float peak_coeff = (random) ? 1.0 : value;
    if (not normalize)
        peak_coeff *= inv_sqrt_2pi / std_dev;

    for (int row = 0 ; row < gauss_rows; ++row) {
        for (int col = 0 ; col < gauss_columns; ++col) {
            int index = row * gauss_columns + col;
            int d_row = row - row_center;
            int d_col = col - column_center;
            float dist = std::sqrt((d_row * d_row) + (d_col * d_col));

            float a = dist / std_dev;
            gaussians[index] = peak_coeff * std::exp(-0.5f * a * a);
        }
    }

    this->update();
}

GaussianRandomInputModule::~GaussianRandomInputModule() {
    this->gaussians.free();
}

void GaussianRandomInputModule::update() {
    fSet(values, values.get_size(), 0.0);

    for (int i = 0 ; i < num_peaks ; ++i) {
        // Randomly select gaussian center
        int row_offset = iRand(0, rows-1);
        int column_offset = iRand(0, columns-1);

        if (random) {
            float peak = fRand(min_value, max_value);
            for (int row = 0 ; row < rows; ++row) {
                for (int col = 0 ; col < columns; ++col) {
                    int index = row * columns + col;
                    int gauss_index =
                        ((row + row_offset) * gauss_columns)
                            + (col + column_offset);
                    values[index] += peak * gaussians[gauss_index];
                }
            }
        } else {
            for (int row = 0 ; row < rows; ++row) {
                for (int col = 0 ; col < columns; ++col) {
                    int index = row * columns + col;
                    int gauss_index =
                        ((row + row_offset) * gauss_columns)
                            + (col + column_offset);
                    values[index] += gaussians[gauss_index];
                }
            }
        }
    }
}
