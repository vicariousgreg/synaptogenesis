#ifndef weight_matrices_h
#define weight_matrices_h

#include "model/model.h"
#include "util/constants.h"

class WeightMatrices {
    public:
        WeightMatrices(Model *model, int weight_depth);
        virtual ~WeightMatrices();

        float* get_matrix(Connection* conn) {
            return this->matrices[conn];
        }

    protected:
        float* matrix_datas;
        std::map<Connection*, float*> matrices;
};

#endif
