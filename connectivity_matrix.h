#pragma once
#include "layer.h"
#include "matrix.h"

class ConnectivityMatrix {
    public:
        ConnectivityMatrix (Layer from_layer, Layer to_layer, bool plastic) :
                from_index(from_layer.start_index),
                to_index(to_layer.start_index),
                from_size(from_layer.size),
                to_size(to_layer.size),
                plastic(plastic),
                sign(from_layer.sign),
                matrix(to_layer.size, from_layer.size) {
            this->matrix.randomize();
        }

        virtual ~ConnectivityMatrix () {}

        int from_index;
        int from_size;
        int to_index;
        int to_size;
        int sign;
        bool plastic;
        Matrix matrix;

    private:
    protected:
};
