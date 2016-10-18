#ifndef random_input_module_h
#define random_input_module_h

#include "io/input_module.h"

class RandomInputModule : public InputModule {
    public:
        RandomInputModule(Layer *layer, std::string params);
        void feed_input(Buffer *buffer);

        float max_value;
};

#endif
