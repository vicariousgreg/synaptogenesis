#ifndef noise_input_module_h
#define noise_input_module_h

#include "io/module/module.h"

class NoiseInputModule : public Module {
    public:
        NoiseInputModule(Layer *layer, std::string params);
        void feed_input(Buffer *buffer);
        virtual IOType get_type() { return INPUT; }

    private:
        float max_value;
};

#endif
