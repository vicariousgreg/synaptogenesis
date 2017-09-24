#ifndef dsst_window_h
#define dsst_window_h

#include "network/layer.h"
#include "util/constants.h"

class DSSTModule;

class DSSTWindow {
    public:
        static DSSTWindow* build(DSSTModule *module);
        virtual void add_layer(Layer *layer, IOTypeMask io_type) = 0;
        virtual void feed_input(Layer *layer, float *input) = 0;
        virtual void input_symbol(int index) = 0;
};

#endif
