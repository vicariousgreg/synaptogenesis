#ifndef connection_h
#define connection_h

#include <vector>

#include "model/connection_config.h"
#include "model/weight_config.h"
#include "util/constants.h"

class Layer;

/* Represents a connection between two neural layers.
 * Connections bridge Layers and are constructed in the Model class.
 * Connections have several types, enumerated and documented in "util/constants.h".
 */
class Connection {
    public:
        virtual ~Connection();

        /* Constant getters */
        int get_num_weights() const;
        const WeightConfig* get_weight_config() const;
        int get_row_field_size() const;
        int get_column_field_size() const;
        int get_row_stride() const;
        int get_column_stride() const;
        int get_row_offset() const;
        int get_column_offset() const;

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

        // Arborization parameters (extracted from params)
        // The receptive field size, stride and offset for arborized
        //   (convergent) connections
        int row_field_size, column_field_size;
        int row_stride, column_stride;
        int row_offset, column_offset;

        // Weight initializer
        const WeightConfig* weight_config;
};

typedef std::vector<Connection*> ConnectionList;

#endif
