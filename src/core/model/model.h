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

        Structure* add_structure(std::string name, std::string engine_name);
        const StructureList& get_structures() const { return structures; }

        // Sum caluclators
        int get_num_neurons() const;
        int get_num_layers() const;
        int get_num_connections() const;
        int get_num_weights() const;

    private:
        // Structures
        StructureList structures;
};

#endif
