#ifndef connection_h
#define connection_h

#include <vector>

#include "model/connection_config.h"
#include "model/weight_config.h"
#include "util/constants.h"

class Layer;

/* Gets the expected row/col size of a destination layer given a |source_layer|,
 *   a connection |type| and connection |params|.
 * This function only returns meaningful values for connection types that
 *   are not FULLY_CONNECTED, because they can link layers of any sizes */
int get_expected_rows(int rows, ConnectionType type, ConnectionConfig config);
int get_expected_columns(int columns, ConnectionType type, ConnectionConfig config);

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
