#ifndef weight_matrix_h
#define weight_matrix_h

#include "model/model.h"
#include "util/constants.h"

class WeightMatrix {
    public:
        WeightMatrix(Connection *conn, int matrix_depth);
        virtual ~WeightMatrix();

        float* get_data() const { return mData; }

    private:
        float *mData;
};

#endif
