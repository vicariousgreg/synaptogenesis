#ifndef attribute_data_h
#define attribute_data_h

#include "util/pointer.h"

class Layer;
class State;
class Attributes;

/* Data package that is passed into synaptic kernel functions */
class AttributeData {
    public:
        AttributeData(Layer *layer, State *state);

        const Attributes *attributes;

        /* IO pointers */
        Pointer<float> input;
        Pointer<Output> output;
        Pointer<Output> expected;

        /* Layer properties */
        const int size;
        const int num_weights;
        int history_size;
        bool plastic;
};

#endif
