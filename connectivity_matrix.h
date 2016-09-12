#ifndef connectivity_matrix_h
#define connectivity_matrix_h

#include "layer.h"
#include "matrix.h"

class ConnectivityMatrix {
    public:
        ConnectivityMatrix (Layer from_layer, Layer to_layer, bool plastic);

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

#endif
