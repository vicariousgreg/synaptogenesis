#ifndef spike_print_output_h
#define spike_print_output_h

#include "io/output.h"
#include "tools.h"

class SpikePrintOutput : public Output {
    public:
        SpikePrintOutput(Layer *layer, std::string params);
        void report_output(State *state);

    int history_length;
    int step_size;
    int counter;
    int refresh_rate;
    unsigned int maximum;
    unsigned int* reverses;

    Timer timer;
};

#endif
