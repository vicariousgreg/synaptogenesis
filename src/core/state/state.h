#ifndef state_h
#define state_h

#include <map>
#include <set>
#include <vector>
#include <string>

#include "state/attributes.h"
#include "util/constants.h"
#include "util/pointer.h"

class Buffer;
class Model;
class Layer;
class DendriticNode;

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Transfers all data to device or back to host */
        void transfer_to_device();
        void transfer_to_host();

        /* Save or load state to/from disk */
        static bool exists(std::string file_name);
        void save(std::string file_name, bool verbose=false);
        void load(std::string file_name, bool verbose=false);

        /* Checks if a structure's layers are compatible with its stream type */
        bool check_compatibility(Structure *structure);

        /* Getters for layer related data */
        DeviceID get_device_id(Layer *layer) const;
        int get_layer_index(Layer *layer) const;
        int get_other_start_index(Layer *layer) const;
        Pointer<float> get_input(Layer *layer, int register_index = 0) const;
        Pointer<float> get_second_order_weights(DendriticNode *node) const;
        Pointer<Output> get_expected(Layer *layer) const;
        Pointer<Output> get_output(Layer *layer, int word_index = 0) const;
        Pointer<float> get_buffer_input(Layer *layer) const;
        Pointer<Output> get_buffer_expected(Layer *layer) const;
        Pointer<Output> get_buffer_output(Layer *layer) const;
        OutputType get_output_type(Layer *layer) const;
        const Attributes *get_attributes_pointer(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> get_attribute_kernel(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> get_learning_kernel(Layer *layer) const;

        /* Getters for connection related data */
        int get_connection_index(Connection *conn) const;
        Pointer<float> get_matrix(Connection *conn) const;
        EXTRACTOR get_extractor(Connection *conn) const;
        Kernel<SYNAPSE_ARGS> get_activator(Connection *conn) const;
        Kernel<SYNAPSE_ARGS> get_updater(Connection *conn) const;
        Pointer<Output> get_device_output_buffer(
            Connection *conn, int word_index) const;
        bool is_inter_device(Connection *conn) const;

        Model* const model;

    private:
        int num_devices;
        std::vector<Buffer*> internal_buffers;
        std::vector<std::map<int, Buffer*> > inter_device_buffers;
        std::vector<std::map<std::string, Attributes*> > attributes;
        std::map<Connection*, WeightMatrix*> weight_matrices;
        std::map<Layer*, DeviceID> layer_devices;

        // Keep track of all pointers
        std::vector<std::vector<BasePointer*> > network_pointers;
        std::vector<std::vector<BasePointer*> > buffer_pointers;

        std::map<PointerKey, BasePointer*> pointer_map;
};

#endif
