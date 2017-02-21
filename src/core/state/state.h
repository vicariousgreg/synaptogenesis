#ifndef state_h
#define state_h

#include "model/model.h"
#include "io/buffer.h"
#include "engine/kernel/kernel.h"
#include "state/attributes.h"
#include "state/weight_matrix.h"
#include "util/constants.h"

class Engine;

class State {
    public:
        State(Model *model);
        virtual ~State();

        /* Builds an engine based on attribute requirements */
        Engine *build_engine() {
            return attributes->build_engine(model, this);
        }

        /* Getters for weight matrices */
        float* get_matrix(Connection* conn) const {
            return weight_matrices.at(conn)->get_data();
        }

        /* Getters for IO data */
        float* get_input(Layer *layer) const { return attributes->get_input(layer->id); }
        OutputType get_output_type() const { return attributes->output_type; }
        Output* get_output(Layer *layer, int word_index = 0) const {
            return attributes->get_output(layer->id, word_index);
        }

        /* Getters for neuron count related information */
        int get_num_neurons() const { return attributes->total_neurons; }
        int get_start_index(Layer *layer) const {
            return attributes->get_start_index(layer->id);
        }

        /* Constant getter so that nobody else changes the Attributes
         * This way, kernels can access attribute data without using a getter
         *     function, but the data is protected from everybody but this State */
        const Attributes *get_attributes_pointer() const { return attributes->pointer; }
        Buffer *get_buffer() const { return buffer; }
        KERNEL get_activator(ConnectionType type) const {
            return attributes->get_activator(type);
        }
        KERNEL get_updater(ConnectionType type) const {
            return attributes->get_updater(type);
        }
        const ATTRIBUTE_KERNEL get_attribute_kernel() const {
            return attributes->get_attribute_kernel();
        }

#ifdef PARALLEL
        cudaStream_t io_stream;
#endif

    private:
        Model *model;
        Attributes *attributes;
        Buffer *buffer;
        std::map<Connection*, WeightMatrix*> weight_matrices;
};

#endif
