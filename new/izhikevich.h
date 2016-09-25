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
        void set_current(int offset, int size, float* input);
        void randomize_current(int offset, int size, float max);
        void clear_current(int offset, int size);
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
        // Locations to store local copies of spikes and currents.
        // When accessed, these values will be copied here from the device.
        int* local_spikes;
        float* local_current;
#endif

};

#endif
