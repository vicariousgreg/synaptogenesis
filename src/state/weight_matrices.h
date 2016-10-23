#ifndef weight_matrices_h
#define weight_matrices_h

#include "model/model.h"
#include "constants.h"

class WeightMatrices {
    public:
        WeightMatrices(Model *model, int weight_depth);
        virtual ~WeightMatrices() {
#ifdef PARALLEL
            cudaFree(this->matrices[0]);
#else
            free(this->matrices[0]);
#endif
            // This is on the host regardless
            free(this->matrices);
        }


        float* get_matrix(int connection_id) {
            return this->matrices[connection_id];
        }

    protected:
        float** matrices;
};

#endif
