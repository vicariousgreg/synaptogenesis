#ifndef connection_h
#define connection_h

#include <vector>

#include "network/connection_config.h"
#include "network/weight_config.h"
#include "util/constants.h"

class Layer;
class DendriticNode;

/* Represents a connection between two neural layers.
 * Connections bridge Layers and are constructed in the Network class.
 * Connections have several types, enumerated and documented in "util/constants.h".
 */
class Connection {
    public:
        virtual ~Connection();

        /* Constant getters */
        int get_num_weights() const;
        int get_compute_weights() const;
        const ConnectionConfig* get_config() const;
        std::string get_parameter(
            std::string key, std::string default_val) const;

        // Connection ID
        const size_t id;

        // Matrix type
        const ConnectionType type;

        // Convolutional boolean (extracted from type)
        const bool convolutional;

        // Second order flag
        const bool second_order;
        const bool second_order_host;
        const bool second_order_slave;

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

        std::string str() const;

    private:
        friend class Structure;

        Connection(Layer *from_layer, Layer *to_layer,
            ConnectionConfig *config, DendriticNode *node);

        // Number of weights in connection
        int num_weights;

        // Connection config
        const ConnectionConfig* config;
};

typedef std::vector<Connection*> ConnectionList;

#endif
