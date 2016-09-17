#include "connectivity_matrix.h"
#include "layer.h"

ConnectivityMatrix::ConnectivityMatrix (Layer from_layer, Layer to_layer,
        bool plastic, double max_weight) :
            from_index(from_layer.start_index),
            to_index(to_layer.start_index),
            from_size(from_layer.size),
            to_size(to_layer.size),
            plastic(plastic),
            sign(from_layer.sign),
            matrix(to_layer.size, from_layer.size) {
    this->matrix.randomize(true, max_weight);
}
