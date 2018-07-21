#ifndef buffer_h
#define buffer_h

#include <vector>
#include <set>
#include <map>

#include "network/layer.h"
#include "util/constants.h"
#include "util/resources/pointer.h"

class Network;

typedef std::map<Layer*, KeySet> LayerKeyMap;

class Buffer {
    public:
        Buffer(DeviceID device_id,
            LayerList input_layers, LayerList output_layers,
            LayerKeyMap input_keys = {},
            LayerKeyMap output_keys = {});
        virtual ~Buffer();

        std::vector<BasePointer*> get_pointers();

        /* IO setters */
        void set_input(Layer *layer, Pointer<float> source);
        void set_output(Layer *layer, Pointer<Output> source);

        /* IO getters */
        Pointer<float> get_input(Layer *layer);
        Pointer<Output> get_output(Layer *layer);
        BasePointer* get_input_auxiliary(Layer *layer, std::string key);
        BasePointer* get_output_auxiliary(Layer *layer, std::string key);

        /* Dirty */
        bool get_input_dirty(Layer *layer) const;
        bool set_input_dirty(Layer *layer, bool dirty=true);
        bool get_auxiliary_dirty(Layer *layer, std::string key) const;
        bool set_auxiliary_dirty(Layer *layer, std::string key, bool dirty=true);

        const DeviceID device_id;

    protected:
        // Buffer data
        std::map<Layer*, Pointer<float>*> input;
        std::map<Layer*, Pointer<Output>*> output;
        std::map<Layer*, std::map<std::string, BasePointer*>> input_auxiliary;
        std::map<Layer*, std::map<std::string, BasePointer*>> output_auxiliary;

        // Dirty maps for input data
        std::map<Layer*, bool> input_dirty_map;
        std::map<Layer*, std::map<std::string, bool>> auxiliary_dirty_map;
};

Buffer *build_buffer(DeviceID device_id,
    LayerList input_layers, LayerList output_layers,
    LayerKeyMap input_keys = {},
    LayerKeyMap output_keys = {});

#endif
