#ifndef rate_encoding_driver_h
#define rate_encoding_driver_h

#include "driver.h"
#include "model.h"

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Rate Encoding model */
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

        // SETTERS
        /* If parallel, these will copy data to the device */
        void set_input(int layer_id, float* input);
        void randomize_input(int layer_id, float max);
        void clear_input(int layer_id);

        // GETTERS
        float* get_input();

        //////////////////////
        /// MODEL SPECIFIC ///
        //////////////////////
        //
        // GETTERS
        /* If parallel, these will copy data from the device */
        float* get_output();

    private:
        // Neuron Spikes
        float* output;

        // Neuron parameters
        RateEncodingParameters *neuron_parameters;

#ifdef PARALLEL
        // Locations to store device copies of data.
        // When accessed, these values will be copied here from the device.
        float *device_output;
        RateEncodingParameters* device_neuron_parameters;
#endif

};

#endif
