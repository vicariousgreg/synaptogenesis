#include "layer.h"
#include "weight_matrix.h"

Layer::Layer(int start_index, int size, int sign) :
        index(start_index),
        size(size),
        sign(sign) {}

void Layer::add_incoming_connection(WeightMatrix &matrix) {
    this->incoming_connections.push_back(&matrix);
}
