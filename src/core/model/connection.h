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
        const ConnectionConfig* get_config() const;
        std::string get_parameter(
            std::string key, std::string default_val) const;

        // Connection ID
        const int id;

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

        // Global counter for ID assignment
        static int count;

        Connection(Layer *from_layer, Layer *to_layer, ConnectionConfig *config);

        // Number of weights in connection
        int num_weights;

        // Connection config
        const ConnectionConfig* config;
};

typedef std::vector<Connection*> ConnectionList;

#endif
