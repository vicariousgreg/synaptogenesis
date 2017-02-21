#ifndef model_h
#define model_h

#include <map>
#include <string>

#include "util/constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "model/structure.h"

/* Represents a full neural network model.
 *
 * Models are built up using Structures, which are subgraphs of the full
 *     network.  They can be built individually and added to the model.
 */
class Model {
    public:
        Model (std::string engine_name);
        virtual ~Model();

        /* TODO: Loads a model from a file using the model builder */
        static Model* load(std::string path);

        /* Adds a structure to the model */
        void add_structure(Structure *structure);

        /* Gets the total neuron count */
        int get_num_neurons() const { return total_neurons; }
        int get_num_neurons(IOType type) const { return num_neurons[type]; }

        /* Getters for constant vector references */
        const ConnectionList& get_connections() const { return connections; }
        const LayerList& get_layers() const { return all_layers; }
        const LayerList& get_layers(IOType type) const { return layers[type]; }

        /* Extracts layers and connections from structures.
         * Reorganizes the model to place input and ouput layers in
         *   contiguous memory. */
        void build();

        // Engine string indicating type of engine
        const std::string engine_name;

    private:
        // Map of names to structures
        std::map<std::string, Structure*> structures;

        // Layers
        LayerList layers[sizeof(IOTypes)];
        LayerList all_layers;

        // Connections
        ConnectionList connections;

        // Total number of neurons
        int total_neurons;
        int num_neurons[sizeof(IOTypes)];
};

#endif
