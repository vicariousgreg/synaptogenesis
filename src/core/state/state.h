#ifndef state_h
#define state_h

#include "model/model.h"
#include "model/layer.h"
#include "io/environment.h"
#include "engine/kernel/synapse_kernel.h"
#include "engine/stream/stream.h"
#include "state/attributes.h"
#include "state/weight_matrix.h"
#include "util/constants.h"
#include "util/pointer.h"

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Checks if a structure's layers are compatible with its stream type */
        bool check_compatibility(Structure *structure);

        /* Getters for layer related data */
        int get_other_start_index(Layer *layer) const;
        Pointer<float> get_input(Layer *layer, int register_index = 0) const;
        Pointer<Output> get_output(Layer *layer, int word_index = 0) const;
        OutputType get_output_type(Layer *layer) const;
        const Attributes *get_attributes_pointer(Layer *layer) const;
        const ATTRIBUTE_KERNEL get_attribute_kernel(Layer *layer) const;

        /* Getters for connection related data */
        Pointer<float> get_matrix(Connection *conn) const;
        EXTRACTOR get_extractor(Connection *conn) const;
        SYNAPSE_KERNEL get_activator(Connection *conn) const;
        SYNAPSE_KERNEL get_updater(Connection *conn) const;

        Model* const model;

    private:
        Attributes* attributes[sizeof(NeuralModels)];
        std::map<Connection*, WeightMatrix*> weight_matrices;
};

#endif
