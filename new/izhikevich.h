#ifndef izhikevich_h
#define izhikevich_h

#include "driver.h"
#include "model.h"

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class IzhikevichParameters {
    public:
        IzhikevichParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}

        float a;
        float b;
        float c;
        float d;
};

class Izhikevich : public Driver {
    public:
        void build(Model* model);

        void step_input();
        void step_state();
        void step_output();
        void step_weights();

        //////////////////////
        /// MODEL SPECIFIC ///
        //////////////////////
        //
        // SETTERS
        /* If parallel, these will copy data to the device */
        void set_current(int layer_id, float* input);
        void randomize_current(int layer_id, float max);
        void clear_current(int layer_id);
        //
        // GETTERS
        /* If parallel, these will copy data from the device */
        int* get_spikes();
        float* get_current();
        float* get_voltage();
        float* get_recovery();

    private:
        // Network model
        Model* model;

        // Neuron States
        float *current;
        float *voltage;
        float *recovery;

        // Neuron Spikes
        int* spikes;
        // Recent points to the most recent integers for spike bit vectors
        int* recent_spikes;

        // Neuron parameters
        IzhikevichParameters *neuron_parameters;

        // Weight matrices
        float** weight_matrices;

#ifdef PARALLEL
        // Locations to store device copies of data.
        // When accessed, these values will be copied here from the device.
        int *device_spikes;
        float *device_current;
        float *device_voltage;
        float *device_recovery;
        int* device_recent_spikes;
        IzhikevichParameters* device_neuron_parameters;
#endif

};

#endif
