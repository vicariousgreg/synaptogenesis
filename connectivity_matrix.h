#ifndef connectivity_matrix_h
#define connectivity_matrix_h

#include "layer.h"

class ConnectivityMatrix {
    public:
        ConnectivityMatrix (Layer from_layer, Layer to_layer,
            bool plastic, double max_weight);

        void build();

        virtual ~ConnectivityMatrix () {}

        int from_index;
        int from_size;
        int to_index;
        int to_size;
        int sign;
        bool plastic;
        double max_weight;
        double* mData;

        void randomize(bool self_connected, double max_weight);

        double& operator()(int i, int j) {
            return this->mData[i * to_size + j];
        }

        double operator()(int i, int j) const {
            return this->mData[i * to_size + j];
        }

};

#endif
