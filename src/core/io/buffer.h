#ifndef buffer_h
#define buffer_h

#include <map>

#include "model/layer.h"
#include "util/constants.h"

class Structure;

class Buffer {
    public:
        Buffer(Structure *structure);
        Buffer(LayerList layers);
        Buffer(LayerList input_layers, LayerList output_layers);
        virtual ~Buffer();

        /* IO setters */
        void set_input(Layer *layer, float* source);
        void set_output(Layer *layer, Output* source);

        /* IO getters */
        float* get_input(Layer *layer) { return input_map[layer]; }
        Output* get_output(Layer *layer) { return output_map[layer]; }

    private:
        void init(LayerList input_layers, LayerList output_layers);

        float *input;
        Output *output;

        int input_size;
        int output_size;

        std::map<Layer*, float*> input_map;
        std::map<Layer*, Output*> output_map;
};

#endif
