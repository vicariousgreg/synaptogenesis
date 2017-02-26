#ifndef state_h
#define state_h

#include "model/model.h"
#include "model/structure.h"
#include "io/environment.h"
#include "engine/kernel/synapse_kernel.h"
#include "state/attributes.h"
#include "state/weight_matrix.h"
#include "util/constants.h"
#include "util/pointer.h"

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Builds an engine based on attribute requirements */
        virtual std::string get_stream_cluster_name(Structure *structure);

        /* Getters for layer related data */
        int get_input_start_index(Layer *layer) const;
        int get_output_start_index(Layer *layer) const;
        int get_other_start_index(Layer *layer) const;
        Pointer<float> get_input(Layer *layer, int register_index = 0) const;
        Pointer<Output> get_output(Layer *layer, int word_index = 0) const;
        const Attributes *get_attributes_pointer(Layer *layer) const;
        const ATTRIBUTE_KERNEL get_attribute_kernel(Layer *layer) const;

        /* Getters for connection related data */
        Pointer<float> get_matrix(Connection *conn) const;
        EXTRACTOR get_extractor(Connection *conn) const;
        SYNAPSE_KERNEL get_activator(Connection *conn) const;
        SYNAPSE_KERNEL get_updater(Connection *conn) const;

        /* Getters for structure related data */
        OutputType get_output_type(Structure *structure) const;

#ifdef PARALLEL
        cudaStream_t io_stream;
#endif

        Model* const model;

    private:
        std::map<Structure*, Attributes*> attributes;
        std::map<Connection*, WeightMatrix*> weight_matrices;
};

#endif
