#ifndef connection_h
#define connection_h

#include <vector>

#include "network/connection_config.h"
#include "util/constants.h"

class Layer;
class DendriticNode;

/* Represents a connection between two neural layers.
 * Connections bridge Layers and are constructed in the Network class.
 * Connections have several types, enumerated and documented in "util/constants.h".
 */
class Connection {
    public:
        Connection(const Connection& other);
        virtual ~Connection();

        /* Constant getters */
        int get_num_weights() const;
        int get_compute_weights() const;
        const ConnectionConfig* get_config() const;
        std::string get_parameter(
            std::string key, std::string default_val) const;

        // Connection config
        const ConnectionConfig * const config;

        // Connected layers
        Layer* const from_layer;
        Layer* const to_layer;
        DendriticNode* const node;

        // Flag for whether matrix can change via learning
        const bool plastic;

        // Connection delay
        const int delay;

        // Maximum for weights
        const float max_weight;

        // Connection operation code
        const Opcode opcode;

        // Matrix type
        const ConnectionType type;

        // Convolutional boolean (extracted from type)
        const bool convolutional;

        // Second order flag
        const bool second_order;
        const bool second_order_host;
        const bool second_order_slave;

        // Optional connection name
        const std::string name;

        // Connection ID
        const size_t id;

        // Number of weights
        const int num_weights;

        std::string str() const;

    protected:
        friend class Structure;

        Connection(Layer *from_layer, Layer *to_layer,
            const ConnectionConfig *config);
};

typedef std::vector<Connection*> ConnectionList;

#endif
