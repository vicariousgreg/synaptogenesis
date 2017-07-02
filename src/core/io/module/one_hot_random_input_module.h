#ifndef one_hot_random_input_module_h
#define one_hot_random_input_module_h

#include "io/module/module.h"

class OneHotRandomInputModule : public Module {
    public:
        OneHotRandomInputModule(Layer *layer, ModuleConfig *config);
        virtual ~OneHotRandomInputModule();

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        int timesteps;
        int end;
        int shuffle_rate;
        float max_value;
        float *random_values;

    MODULE_MEMBERS
};

#endif
