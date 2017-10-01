#ifndef network_h
#define network_h

#include <vector>
#include <string>

#include "util/constants.h"
#include "network/layer.h"
#include "network/connection.h"
#include "network/structure.h"
#include "network/network_config.h"

/* Represents a full neural network.
 *
 * Networks are built up using Structures, which are subgraphs of the network.
 */
class Network {
    public:
        Network() : config(new NetworkConfig()) { }
        Network(NetworkConfig* config);
        virtual ~Network();

        /* Save or load model to/from JSON file */
        static Network* load(std::string path);
        void save(std::string path);

        /* Add or retrieve structure to/from model */
        void add_structure(StructureConfig *struct_config);
        Structure* get_structure(std::string name, bool log_error=true);

        /* Add connection */
        void add_connection(const ConnectionConfig* conn_config);

        /* Getters */
        const NetworkConfig* get_config() const { return config; }
        const StructureList& get_structures() const { return structures; }
        const LayerList get_layers() const;
        const LayerList get_layers(std::string neural_model) const;
        const ConnectionList get_connections() const;

        int get_num_neurons() const;
        int get_num_layers() const;
        int get_num_connections() const;
        int get_num_weights() const;
        int get_max_layer_size() const;

    private:
        void add_structure_internal(StructureConfig *struct_config);
        void add_connection_internal(const ConnectionConfig* conn_config);

        NetworkConfig * const config;
        StructureList structures;

};

#endif
