#ifndef neuron_parameters_h
#define neuron_parameters_h

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class NeuronParameters {
    public:
        NeuronParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}

        float a;
        float b;
        float c;
        float d;
};

#endif
