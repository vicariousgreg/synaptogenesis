#include "layer.h"
#include "weight_matrix.h"

Layer::Layer(int start_index, int size) :
        index(start_index),
        size(size) {}

void Layer::add_incoming_connection(WeightMatrix* matrix) {
    this->incoming_connections.push_back(matrix);
}
