#ifndef random_input_module_h
#define random_input_module_h

#include "io/module/module.h"

class RandomInputModule : public Module {
    public:
        RandomInputModule(Layer *layer, ModuleConfig *config);
        virtual ~RandomInputModule();

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        int timesteps;
        int shuffle_rate;
        float max_value;
        float *random_values;

    MODULE_MEMBERS
};

#endif
