#ifndef buffer_h
#define buffer_h

#include <vector>
#include <map>

#include "network/layer.h"
#include "util/constants.h"
#include "util/pointer.h"

class Network;

class Buffer {
    public:
        Buffer(LayerList input_layers, LayerList output_layers,
            LayerList expected_layers, DeviceID device_id);
        virtual ~Buffer();

        std::vector<BasePointer*> get_pointers();

        /* IO setters */
        void set_input(Layer *layer, Pointer<float> source);
        void set_output(Layer *layer, Pointer<Output> source);
        void set_expected(Layer *layer, Pointer<Output> source);

        /* IO getters */
        Pointer<float> get_input(Layer *layer);
        Pointer<Output> get_output(Layer *layer);
        Pointer<Output> get_expected(Layer *layer);

        /* Dirty */
        bool get_input_dirty(Layer *layer) const;
        bool set_input_dirty(Layer *layer, bool dirty=true);
        bool get_expected_dirty(Layer *layer) const;
        bool set_expected_dirty(Layer *layer, bool dirty=true);

        const DeviceID device_id;

    protected:
        Pointer<float> input;
        Pointer<Output> output;
        Pointer<Output> expected;

        int input_size;
        int output_size;
        int expected_size;

        std::map<Layer*, bool> input_dirty_map;
        std::map<Layer*, bool> expected_dirty_map;
        std::map<Layer*, int> input_map;
        std::map<Layer*, int> output_map;
        std::map<Layer*, int> expected_map;
};

Buffer *build_buffer(DeviceID device_id,
    LayerList input_layers, LayerList output_layers, LayerList expected_layers);

#endif
