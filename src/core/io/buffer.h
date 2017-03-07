#ifndef buffer_h
#define buffer_h

#include <map>

#include "model/layer.h"
#include "util/constants.h"
#include "util/pointer.h"

class Model;

class Buffer {
    public:
        Buffer(Model *model);
        Buffer(LayerList input_layers, LayerList output_layers);
        virtual ~Buffer();

        /* IO setters */
        void set_input(Layer *layer, Pointer<float> source);
        void set_output(Layer *layer, Pointer<Output> source);

        /* IO getters */
        Pointer<float> get_input(Layer *layer) { return input_map[layer]; }
        Pointer<Output> get_output(Layer *layer) { return output_map[layer]; }

        /* Dirty */
        bool get_dirty(Layer *layer) const { return dirty_map.at(layer); }
        bool set_dirty(Layer *layer, bool dirty=true) {
            dirty_map[layer] = dirty;
        }

    protected:
        virtual void init();

        Pointer<float> input;
        Pointer<Output> output;

        LayerList input_layers;
        LayerList output_layers;

        int input_size;
        int output_size;

        std::map<Layer*, bool> dirty_map;
        std::map<Layer*, Pointer<float> > input_map;
        std::map<Layer*, Pointer<Output> > output_map;
};

class HostBuffer : public Buffer {
    public:
        HostBuffer(Model *model) : Buffer(model) { init(); }
        HostBuffer(LayerList input_layers, LayerList output_layers)
                : Buffer(input_layers, output_layers) { init(); }

    protected:
        virtual void init();
};

class DeviceBuffer : public Buffer {
    public:
        DeviceBuffer(Model *model) : Buffer(model) { init(); }
        DeviceBuffer(LayerList input_layers, LayerList output_layers)
                : Buffer(input_layers, output_layers) { init(); }

    protected:
        virtual void init();
};

#endif
