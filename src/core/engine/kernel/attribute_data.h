#ifndef attribute_data_h
#define attribute_data_h

#include "model/layer.h"
#include "util/parallel.h"

class State;
class Attributes;

/* Data package that is passed into synaptic kernel functions */
class AttributeData {
    public:
        AttributeData(Layer *layer, State *state);
        const Attributes *attributes;
        const int input_start_index;
        const int output_start_index;
        const int other_start_index;
        const int size;
        int history_size;
};

#endif
