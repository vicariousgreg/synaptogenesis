#ifndef saccade_window_h
#define saccade_window_h

#include "network/layer.h"
#include "util/constants.h"

class SaccadeModule;

class SaccadeWindow {
    public:
        static SaccadeWindow* build(SaccadeModule *module);
        virtual void lock() = 0;
        virtual void unlock() = 0;
        virtual void add_layer(Layer *layer, IOTypeMask io_type) = 0;
        virtual void prepare_input_data() = 0;
        virtual void feed_input(Layer *layer, float *input) = 0;
        virtual void feed_central_input(Layer *layer, float *input) = 0;
        virtual void report_output(Layer *layer,
            Output *output, OutputType output_type) = 0;
};

#endif
