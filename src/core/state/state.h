#ifndef state_h
#define state_h

#include <map>
#include <set>
#include <vector>
#include <string>

#include "state/attributes.h"
#include "util/constants.h"
#include "util/resources/pointer.h"

class Buffer;
class Network;
class Layer;
class DendriticNode;

typedef std::map<DeviceID, std::vector<BasePointer*>> PointerSetMap;
typedef std::map<PointerKey, BasePointer*> PointerMap;

class State {
    public:
        State(Network *network);
        virtual ~State();

        /* Builds the state on the specified devices
         * If none specified, host is used by default */
        void build(std::set<DeviceID> devices = {});
        const std::set<DeviceID> get_active_devices() const
            { return active_devices; }

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

        /* Getters for layer related data */
        DeviceID get_device_id(Layer *layer) const;
        Pointer<float> get_input(Layer *layer, int register_index = 0) const;
        Pointer<float> get_second_order_weights(DendriticNode *node) const;
        Pointer<Output> get_output(Layer *layer, int word_index = 0) const;
        Pointer<float> get_buffer_input(Layer *layer) const;
        const Attributes *get_attributes_pointer(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> get_attribute_kernel(Layer *layer) const;
        Kernel<ATTRIBUTE_ARGS> get_learning_kernel(Layer *layer) const;

        /* Getters for connection related data */
        Pointer<float> get_weights(Connection *conn) const;
        const WeightMatrix* get_matrix(Connection *conn) const;
        const WeightMatrix* get_matrix_pointer(Connection *conn) const;
        EXTRACTOR get_connection_extractor(Connection *conn) const;
        AGGREGATOR get_connection_aggregator(Connection *conn) const;
        KernelList<SYNAPSE_ARGS> get_activators(Connection *conn) const;
        KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn) const;
        KeySet get_init_keys(Layer *layer) const;
        Pointer<Output> get_device_output_buffer(
            Connection *conn, int word_index) const;
        bool is_inter_device(Connection *conn) const;
        bool get_transpose_flag(Connection *conn) const;

        /* Getters for external use */
        BasePointer* get_neuron_data(Layer *layer, std::string key);
        BasePointer* get_layer_data(Layer *layer, std::string key);
        BasePointer* get_connection_data(Connection *conn, std::string key);
        BasePointer* get_weight_matrix(Connection *conn,
            std::string key="weights");

        Network* const network;

    private:
        // Flag for whether state data is on host
        bool on_host;

        // Data buffers
        std::map<DeviceID, Buffer*> internal_buffers;
        std::map<DeviceID, std::map<int, Buffer*>> inter_device_buffers;

        // Layer maps
        std::map<Layer*, Attributes*> attributes;
        std::map<Layer*, DeviceID> layer_devices;
        std::set<DeviceID> active_devices;

        // Functions for gathering pointers
        PointerSetMap get_network_pointers() const;
        PointerSetMap get_buffer_pointers() const;
        PointerMap get_pointer_map() const;
};

#endif
