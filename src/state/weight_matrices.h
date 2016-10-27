#ifndef weight_matrices_h
#define weight_matrices_h

#include "model/model.h"
#include "util/constants.h"

class WeightMatrices {
    public:
        WeightMatrices(Model *model, int weight_depth);
        virtual ~WeightMatrices();


        float* get_matrix(int connection_id) {
            return this->matrices[connection_id];
        }

    protected:
        float** matrices;
};

#endif
