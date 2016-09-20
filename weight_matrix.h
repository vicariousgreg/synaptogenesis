#ifndef weight_matrix_h
#define weight_matrix_h

#include "layer.h"

class WeightMatrix {
    public:
        WeightMatrix (Layer from_layer, Layer to_layer,
            bool plastic, float max_weight);

        void build();

        virtual ~WeightMatrix () {}

        int from_index, from_size;
        int to_index, to_size;
        int sign;
        bool plastic;

        float max_weight;
        float* mData;

        void randomize(bool self_connected, float max_weight);

        float& operator()(int i, int j) {
            return this->mData[i * to_size + j];
        }

        float operator()(int i, int j) const {
            return this->mData[i * to_size + j];
        }

};

#endif
