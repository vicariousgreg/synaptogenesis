#ifndef model_h
#define model_h

#include <vector>
#include <string>

#include "util/constants.h"
#include "model/layer.h"
#include "model/connection.h"
#include "model/structure.h"

/* Represents a full neural network model.
 *
 * Models are built up using Structures, which are subgraphs of the full
 *     network.
 */
class Model {
    public:
        virtual ~Model();

        /* TODO: Loads a model from a file using the model builder */
        static Model* load(std::string path);

        void add_structure(Structure *structure);
        Structure* get_structure(std::string name);

        /* Connects layers in two different structures */
        static Connection* connect(
            Structure *from_structure, std::string from_layer_name,
            Structure *to_structure, std::string to_layer_name,
            ConnectionConfig *config);

        const StructureList& get_structures() const { return structures; }
        const LayerList get_layers() const;
        const LayerList get_layers(std::string neural_model) const;
        const LayerList get_input_layers() const;
        const LayerList get_output_layers() const;
        const LayerList get_expected_layers() const;

        int get_num_neurons() const;
        int get_num_layers() const;
        int get_num_connections() const;
        int get_num_weights() const;
        int get_max_layer_size() const;

    private:
        // Structures
        StructureList structures;
};

#endif
