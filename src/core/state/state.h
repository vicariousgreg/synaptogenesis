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
class Network;
class Layer;
class DendriticNode;

class State {
    public:
        State(Network *network);
        virtual ~State();

        /* Transfers all data to device or back to host */
        void transfer_to_device();
        void transfer_to_host();

        /* Copy data over to another state */
        void copy_to(State* other);

        /* Get the size of the state and its buffers */
        size_t get_network_bytes() const;
        size_t get_buffer_bytes() const;

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
        const Attributes *get_attributes_pointer(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> get_attribute_kernel(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> get_learning_kernel(Layer *layer) const;

        /* Getters for connection related data */
        int get_connection_index(Connection *conn) const;
        Pointer<float> get_matrix(Connection *conn) const;
        EXTRACTOR get_connection_extractor(Connection *conn) const;
        Kernel<SYNAPSE_ARGS> get_activator(Connection *conn) const;
        Kernel<SYNAPSE_ARGS> get_updater(Connection *conn) const;
        Pointer<Output> get_device_output_buffer(
            Connection *conn, int word_index) const;
        bool is_inter_device(Connection *conn) const;

        /* Getters for external use */
        BasePointer* get_neuron_data(Layer *layer, std::string key);
        BasePointer* get_layer_data(Layer *layer, std::string key);
        BasePointer* get_connection_data(Connection *conn, std::string key);
        BasePointer* get_weight_matrix(Connection *conn);
        Pointer<float> get_weight_matrix(Connection *conn, int layer);

        Network* const network;

        const std::vector<DeviceID> get_active_devices()
            { return active_devices; }

    private:
        std::vector<DeviceID> active_devices;

        bool on_host;
        std::map<DeviceID, Buffer*> internal_buffers;
        std::map<DeviceID, std::map<int, Buffer*>> inter_device_buffers;
        std::map<DeviceID, std::map<std::string, Attributes*>> attributes;
        std::map<Connection*, WeightMatrix*> weight_matrices;
        std::map<Layer*, DeviceID> layer_devices;

        // Keep track of all pointers
        std::map<DeviceID, std::vector<BasePointer*>> network_pointers;
        std::map<DeviceID, std::vector<BasePointer*>> buffer_pointers;

        std::map<PointerKey, BasePointer*> pointer_map;

        // Pointers to blocks of data returned by ResourceManager
        // Transfers combine all pointers to one block that needs to be freed
        std::set<BasePointer*> data_block_pointers;
};

#endif
