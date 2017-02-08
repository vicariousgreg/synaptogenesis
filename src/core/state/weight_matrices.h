#ifndef weight_matrices_h
#define weight_matrices_h

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

class WeightMatrices {
    public:
        WeightMatrices(Model *model, int matrix_depth);
        virtual ~WeightMatrices();

        float* get_matrix(Connection* conn) const {
            return matrices.at(conn)->get_data();
        }

    protected:
        std::map<Connection*, WeightMatrix*> matrices;
};

#endif
