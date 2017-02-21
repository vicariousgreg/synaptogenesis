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

        /* Primary environment IO functions */
        void transfer_input();
        void transfer_output();

        /* State update functions */
        void update_states();
        void update_states(Layer *layer);

        /* Getters for weight matrices */
        float* get_matrix(Connection* conn) const {
            return weight_matrices.at(conn)->get_data();
        }

        /* Getters for IO data */
        float* get_input() const { return attributes->input; }
        OutputType get_output_type() const { return attributes->output_type; }
        Output* get_recent_output() const { return attributes->recent_output; }
        Output* get_output(int word_index = 0) const {
            return attributes->output + (attributes->total_neurons * word_index);
        }

        /* Getters for neuron count related information */
        int get_num_neurons() const { return attributes->total_neurons; }
        int get_num_neurons(IOType type) const { return attributes->get_num_neurons(type); }
        int get_start_index(IOType type) const { return attributes->get_start_index(type); }

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

#ifdef PARALLEL
        /* If parallel, callers may want to wait for IO events */
        void wait_for_input();
        void wait_for_output();

        /* Cuda streams for IO and state computations */
        cudaStream_t input_stream, output_stream;
        cudaStream_t state_stream;

        /* Cuda events for IO and output events */
        cudaEvent_t
            *input_event,
            *output_event,
            *state_event;
#endif

    private:
        void update_states(int start_index, int count);
        void update_states(IOType layer_type);

        Model *model;
        Attributes *attributes;
        Buffer *buffer;
        std::map<Connection*, WeightMatrix*> weight_matrices;
};

#endif
