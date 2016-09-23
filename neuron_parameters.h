#ifndef neuron_parameters_h
#define neuron_parameters_h

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class NeuronParameters {
    public:
        NeuronParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}

        NeuronParameters copy() {
            return NeuronParameters(this->a, this->b, this->c, this->d);
        }

        float a;
        float b;
        float c;
        float d;
};

#endif
