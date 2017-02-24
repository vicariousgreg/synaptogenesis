#ifndef buffer_h
#define buffer_h

#include <map>

#include "model/layer.h"
#include "util/constants.h"
#include "util/pointer.h"

class Structure;

class Buffer {
    public:
        Buffer(Structure *structure);
        Buffer(LayerList layers);
        Buffer(LayerList input_layers, LayerList output_layers);
        virtual ~Buffer();

        /* IO setters */
        void set_input(Layer *layer, Pointer<float> source);
        void set_output(Layer *layer, Pointer<Output> source);

        /* IO getters */
        Pointer<float> get_input(Layer *layer) { return input_map[layer]; }
        Pointer<Output> get_output(Layer *layer) { return output_map[layer]; }

    private:
        void init(LayerList input_layers, LayerList output_layers);

        Pointer<float> input;
        Pointer<Output> output;

        int input_size;
        int output_size;

        std::map<Layer*, Pointer<float> > input_map;
        std::map<Layer*, Pointer<Output> > output_map;
};

#endif
