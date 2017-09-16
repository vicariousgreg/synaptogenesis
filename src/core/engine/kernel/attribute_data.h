#ifndef attribute_data_h
#define attribute_data_h

#include "model/layer.h"
#include "util/parallel.h"
#include "util/pointer.h"

class State;
class Attributes;

/* Data package that is passed into synaptic kernel functions */
class AttributeData {
    public:
        AttributeData(Layer *layer, State *state);
        const Attributes *attributes;
        Pointer<float> input;
        Pointer<Output> output;
        Pointer<Output> expected;
        const int layer_index;
        const int other_start_index;
        const int size;
        const int num_weights;
        int history_size;
        bool plastic;
};

#endif
