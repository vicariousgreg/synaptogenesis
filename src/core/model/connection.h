#ifndef connection_h
#define connection_h

#include <string>
#include <vector>

#include "model/connection_config.h"
#include "util/constants.h"

class Layer;

/* Gets the expected row/col size of a destination layer given a |source_layer|,
 *   a connection |type| and connection |params|.
 * This function only returns meaningful values for connection types that
 *   are not FULLY_CONNECTED, because they can link layers of any sizes */
int get_expected_rows(int rows, ConnectionType type, std::string params);
int get_expected_columns(int columns, ConnectionType type, std::string params);

/* Represents a connection between two neural layers.
 * Connections bridge Layers and are constructed in the Model class.
 * Connections have several types, enumerated and documented in "util/constants.h".
 */
class Connection {
    public:
        /* Constant getters */
        int get_num_weights() const { return num_weights; }
        const std::string get_init_params() const { return init_params; }
        int get_row_field_size() const { return row_field_size; }
        int get_column_field_size() const { return column_field_size; }
        int get_row_stride() const { return row_stride; }
        int get_column_stride() const { return column_stride; }

        // Matrix type
        const ConnectionType type;

        // Convolutional boolean (extracted from type)
        const bool convolutional;

        // Connected layers
        Layer* const from_layer;
        Layer* const to_layer;

        // Connection delay
        const int delay;

        // Connection operation code
        const Opcode opcode;

        // Flag for whether matrix can change via learning
        const bool plastic;

        // Maximum for weights
        const float max_weight;

    private:
        friend class Structure;

        Connection(Layer *from_layer, Layer *to_layer, ConnectionConfig config);

        // Number of weights in connection
        int num_weights;

        // Parameters for matrix construction
        // Some types will parse values for connection construction
        //   -> Divergent, Convergent, Convolutional
        // In this case, the constructor will consume these values and leave
        //   the remaining values here
        std::string init_params;

        // Arborization parameters (extracted from params)
        // The receptive field size and stride for arborized
        //   (convergent) connections
        int row_field_size, column_field_size;
        int row_stride, column_stride;
};

typedef std::vector<Connection*> ConnectionList;

#endif
