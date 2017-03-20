#ifndef state_h
#define state_h

#include "model/model.h"
#include "model/layer.h"
#include "engine/kernel/synapse_kernel.h"
#include "state/attributes.h"
#include "state/weight_matrix.h"
#include "util/constants.h"
#include "util/pointer.h"

class Buffer;

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Checks if a structure's layers are compatible with its stream type */
        bool check_compatibility(Structure *structure);

        /* Getters for layer related data */
        DeviceID get_device_id(Layer *layer) const;
        int get_other_start_index(Layer *layer) const;
        Pointer<float> get_input(Layer *layer, int register_index = 0) const;
        Pointer<Output> get_expected(Layer *layer) const;
        Pointer<Output> get_output(Layer *layer, int word_index = 0) const;
        Pointer<float> get_buffer_input(Layer *layer) const;
        Pointer<Output> get_buffer_expected(Layer *layer) const;
        Pointer<Output> get_buffer_output(Layer *layer) const;
        OutputType get_output_type(Layer *layer) const;
        const Attributes *get_attributes_pointer(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> const get_attribute_kernel(Layer *layer) const;

        /* Getters for connection related data */
        Pointer<float> get_matrix(Connection *conn) const;
        EXTRACTOR get_extractor(Connection *conn) const;
        Kernel<SYNAPSE_ARGS> get_activator(Connection *conn) const;
        Kernel<SYNAPSE_ARGS> get_updater(Connection *conn) const;
        Pointer<Output> get_device_output_buffer(Connection *conn) const;
        bool is_inter_device(Connection *conn) const;

        Model* const model;

    private:
        int num_devices;
        std::vector<Buffer*> buffers;
        std::vector<std::vector<Attributes*> > attributes;
        std::map<Connection*, WeightMatrix*> weight_matrices;
        std::map<Layer*, DeviceID> layer_devices;
};

#endif
