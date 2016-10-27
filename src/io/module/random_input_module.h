#ifndef random_input_module_h
#define random_input_module_h

#include "io/module/module.h"

class RandomInputModule : public Module {
    public:
        RandomInputModule(Layer *layer, std::string params);
        void feed_input(Buffer *buffer);
        virtual IOType get_type() { return INPUT; }

    private:
        float max_value;
};

#endif
