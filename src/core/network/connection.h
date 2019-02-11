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

        /* Constructs an inverted arborized connection.
         * For use in randomizing divergent projections. */
        static Connection *invert(Connection* other);

        /* Constant getters */
        ConnectionType get_type() const { return type; }
        int get_num_weights() const { return num_weights; }
        int get_compute_weights() const;
        int get_matrix_rows() const;
        int get_matrix_columns() const;
        const ConnectionConfig* get_config() const;
        std::string get_parameter(
            std::string key, std::string default_val) const;

        /* Sparsify functionality */
        void sparsify(int sparse_num_weights);

        /* Gets a parameter from the connection config,
         *   logging a warning if not found */
        std::string str() const;

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

        // Sparse matrix (affects num_weights)
        const bool sparse;

        // Randomized projection
        const bool randomized_projection;

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

    protected:
        friend class Structure;

        // Manual constructor with type override
        // For use with invert method above
        Connection(Connection* other, ConnectionType new_type);

        // Matrix type
        ConnectionType type;

        // Number of weights
        int num_weights;

        Connection(Layer *from_layer, Layer *to_layer,
            const ConnectionConfig *config);
};

typedef std::vector<Connection*> ConnectionList;

#endif
