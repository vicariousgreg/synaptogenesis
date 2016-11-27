#ifndef model_h
#define model_h

#include <vector>
#include <map>
#include <string>

#include "util/constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "model/structure.h"
#include "io/module/module.h"

/* Represents a full neural network model.
 * Contains network graph data and parameters for layers/connections.
 *
 * Models are built up using Structures, which are subgraphs of the full
 *     network.  They can be built individually and added to the model.
 */
class Model {
    public:
        Model (std::string engine_name);
        virtual ~Model() {
            for (auto& conn : this->connections) delete conn;
            for (auto& layer : this->all_layers) delete layer;
            for (auto& structure : this->structures) delete structure.second;
        }

        static Model* load(std::string path);

        void add_structure(Structure *structure);

        /* Extracts layers and connections from structures.
         * Reorganizes the model to place input and ouput layers in
         *   contiguous memory. */
        void build();

        // Engine string indicating type of engine
        std::string engine_name;

        // Total number of neurons
        int num_neurons;

        // Map of names to structures
        std::map<std::string, Structure*> structures;

        // Layers
        std::vector<Layer*> layers[IO_TYPE_SIZE];
        std::vector<Layer*> all_layers;

        // Connections
        std::vector<Connection*> connections;

    private:
        /* Adds a given number of neurons.
         * This is called from add_layer() */
        void add_neurons(int count) {
            this->num_neurons += count;
        }
};

#endif
