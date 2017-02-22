#ifndef state_h
#define state_h

#include "model/model.h"
#include "model/structure.h"
#include "io/buffer.h"
#include "engine/kernel/kernel.h"
#include "state/attributes.h"
#include "state/weight_matrix.h"
#include "util/constants.h"

class StreamCluster;

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Builds an engine based on attribute requirements */
        StreamCluster *build_stream_cluster(Structure *structure);

        /* Getters for layer related data */
        int get_start_index(Layer *layer) const;
        float* get_input(Layer *layer) const;
        OutputType get_output_type(Layer *layer) const;
        Output* get_output(Layer *layer, int word_index = 0) const;
        const Attributes *get_attributes_pointer(Layer *layer) const;
        const ATTRIBUTE_KERNEL get_attribute_kernel(Layer *layer) const;

        /* Getters for connection related data */
        float* get_matrix(Connection* conn) const;
        EXTRACTOR get_extractor(Connection *conn) const;
        KERNEL get_activator(Connection *conn) const;
        KERNEL get_updater(Connection *conn) const;

        /* Getters for structure related data */
        Buffer *get_buffer(Structure *structure) const;
        int get_num_neurons(Structure* structure) const;

#ifdef PARALLEL
        cudaStream_t io_stream;
#endif

    private:
        Model *model;
        std::map<Structure*, Attributes*> attributes;
        std::map<Structure*, Buffer*> buffers;
        std::map<Connection*, WeightMatrix*> weight_matrices;
};

#endif
