#ifndef connection_h
#define connection_h

#include <string>

#include "util/constants.h"

class Layer;

/* Gets the expected row/col size of a destination layer given a |source_layer|,
 *   a connection |type| and connection |params|.
 * This function only returns meaningful values for connection types that
 *   are not FULLY_CONNECTED, because they can link layers of any sizes */
int get_expected_dimension(int source_val, ConnectionType type,
                                            std::string params);

/* Represents a connection between two neural layers.
 * Connections bridge Layers and are constructed in the Model class.
 * Connections have several types, enumerated and documented in "util/constants.h".
 *
 * Connections contain:
 *   - connection type enum
 *   - unique identifier
 *   - parent id if sharing weights with another connection
 *   - convolutional flag if sharing weights internally
 *   - arborization parameters (if applicable)
 *   - parameters for matrix initialization
 *   - total number of actual weights in the connection
 *   - extracted layer properties
 *   - connection delay
 *   - connection opcode (see util/constants.h)
 *   - plasticity boolean
 *   - maximum weight value
 *
 */
class Connection {
    public:
        // Matrix type
        ConnectionType type;

        // Connection ID
        int id;

        // Parent connection if this is a shared connection
        Connection *parent;

        // Convolutional boolean (extracted from type)
        bool convolutional;

        // Arborization parameters (extracted from params)
        // The amount of overlap and stride for arborized
        //   (convergent) connections
        int overlap, stride;

        // Parameters for matrix construction
        // Some types will parse values for connection construction
        //   -> Divergent, Convergent, Convolutional
        // In this case, the constructor will consume these values and leave
        //   the remaining values here
        std::string init_params;

        // Connected layers
        Layer *from_layer, *to_layer;

        // Number of weights in connection
        int num_weights;
        // Connection delay
        int delay;

        // Connection operation code
        Opcode opcode;

        // Flag for whether matrix can change via learning
        bool plastic;

        // Maximum for weights
        float max_weight;

    private:
        friend class Model;
        friend class Structure;

        Connection (int conn_id, Layer *from_layer, Layer *to_layer,
                bool plastic, int delay, float max_weight,
                ConnectionType type, std::string params, Opcode opcode);

        Connection(int conn_id, Layer *from_layer, Layer *to_layer,
                Connection *parent);

};

#endif
