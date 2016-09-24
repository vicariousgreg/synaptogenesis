#ifndef weight_matrix_h
#define weight_matrix_h

#include "layer.h"

/* Matrix Type enumeration.
 * Fully connected represents an n x m matrix.
 * One-to-one represents an n size vector connecting two layers
 *   of idential sizes.
 */
enum MatrixType {
    FULLY_CONNECTED,
    ONE_TO_ONE
};

class WeightMatrix {
    public:
        WeightMatrix (Layer &from_layer, Layer &to_layer,
            bool plastic, float max_weight, MatrixType type);

        /* Allocates memory for the matrix and initializes it.
         * Implementation depends on parallel flag.
         * If parallel, weights are initialized on the device.
         * Returns whether the allocation was successful.
         */
        void build();

        /* Destructor.
         * Frees matrix memory.
         * If parallel, this will free on the device.
         */
        virtual ~WeightMatrix () {}

        /* Randomizes the matrix
         * The bounds for weights are set by |max_weight|.
         *
         * If parallel, a temporary matrix is allocated on the host, which is
         *   initialized and then copied to the device.
         */
        void randomize(float max_weight);

        /* Accessor override */
        float& operator()(int i, int j) {
            return this->mData[i * to_layer.size + j];
        }

        /* Accessor override */
        float operator()(int i, int j) const {
            return this->mData[i * to_layer.size + j];
        }

        // Associated layers
        Layer from_layer, to_layer;

        // Number of weights in matrix.
        int matrix_size;

        // Sign of operation
        // TODO: Replace with function
        int sign;

        // Flag for whether matrix can change via learning
        bool plastic;

        // Maximum weight for randomization
        float max_weight;

        // Pointer to data
        // If parallel, this will point to device memory
        float* mData;

        // Matrix type (see enum)
        MatrixType type;
};

#endif
