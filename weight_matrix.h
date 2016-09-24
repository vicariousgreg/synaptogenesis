#ifndef weight_matrix_h
#define weight_matrix_h

#include "operations.h"
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
        WeightMatrix (Layer &from_layer, Layer &to_layer, bool plastic,
            int delay, float max_weight, MatrixType type, OPCODE opcode);

        /* Allocates memory for the matrix and initializes it.
         * Implementation depends on parallel flag.
         * If parallel, weights are initialized on the device.
         * Returns whether the allocation was successful.
         */
        void build();

        /* Randomizes the matrix
         * The bounds for weights are set by |max_weight|.
         *
         * If parallel, a temporary matrix is allocated on the host, which is
         *   initialized and then copied to the device.
         */
        void randomize(float max_weight);

        // Matrix type (see enum)
        MatrixType type;

        // Associated layers
        Layer from_layer, to_layer;

        // Number of weights in matrix.
        int matrix_size;

        // Connection operation code
        OPCODE opcode;

        // Connection delay
        int delay;

        // Flag for whether matrix can change via learning
        bool plastic;

        // Maximum weight for randomization
        float max_weight;

        // Pointer to data
        // If parallel, this will point to device memory
        float* mData;
};

#endif
