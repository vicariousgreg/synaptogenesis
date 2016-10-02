#ifndef spike_print_output_h
#define spike_print_output_h

#include "../framework/output.h"

class SpikePrintOutput : public Output {
    public:
        SpikePrintOutput(Layer *layer, std::string params);
        void report_output(State *state);
};

#endif
