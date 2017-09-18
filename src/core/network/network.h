#ifndef network_h
#define network_h

#include <vector>
#include <string>

#include "util/constants.h"
#include "network/layer.h"
#include "network/connection.h"
#include "network/structure.h"

/* Represents a full neural network model.
 *
 * Networks are built up using Structures, which are subgraphs of the full
 *     network.
 */
class Network {
    public:
        Network() { }
        virtual ~Network();

        /* Save or load model to/from JSON file */
        static Network* load(std::string path);
        void save(std::string path);

        /* Add or retrieve structure to/from model */
        void add_structure(Structure *structure);
        Structure* get_structure(std::string name);

        /* Getters */
        const StructureList& get_structures() const { return structures; }
        const LayerList get_layers() const;
        const LayerList get_layers(std::string neural_model) const;
        int get_num_neurons() const;
        int get_num_layers() const;
        int get_num_connections() const;
        int get_num_weights() const;
        int get_max_layer_size() const;

    private:
        StructureList structures;
};

#endif
