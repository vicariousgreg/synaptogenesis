#ifndef spiking_attributes_h
#define spiking_attributes_h

#include "state/attributes.h"

class SpikingAttributes : public Attributes {
    public:
        class TraceLearning : public LearningRule {
            public:
                virtual KERNEL get_activator(ConnectionType type);
                virtual KERNEL get_updater(ConnectionType type);
                virtual int get_matrix_depth() { return 3; }
                virtual void process_weight_matrix(WeightMatrix* matrix);
        };

        SpikingAttributes(Model* model);
        ~SpikingAttributes();

        virtual LearningRule *get_learning_rule() {
            if (learning_rule == NULL)
                learning_rule = new TraceLearning();
            return learning_rule;
        }

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        // Neuron Attributes
        float *voltage;

        // Neuron Current (copy of input)
        float* current;

        // Neuron Spikes (copy of output)
        unsigned int* spikes;
};

#endif
