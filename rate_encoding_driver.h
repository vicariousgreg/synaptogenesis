#ifndef rate_encoding_driver_h
#define rate_encoding_driver_h

#include "driver.h"
#include "model.h"

/* Neuron parameters class.
 * Contains parameters for Rate Encoding model */
class RateEncodingParameters {
    public:
        RateEncodingParameters(float x) : x(x) {}
        float x;
};

class RateEncodingDriver : public Driver {
    public:
        void build(Model* model);
        void step_input();
        void step_output();
        void step_weights();

    private:
        // Neuron parameters
        RateEncodingParameters* neuron_parameters;

#ifdef PARALLEL
        // Locations to store device copies of data.
        // When accessed, these values will be copied here from the device.
        RateEncodingParameters* device_neuron_parameters;
#endif

};

#endif
