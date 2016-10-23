#ifndef driver_h
#define driver_h

#include <vector>

#include "state/state.h"
#include "model/model.h"
#include "driver/kernel.h"
#include "driver/instruction.h"
#include "driver/stream.h"
#include "parallel.h"

class Driver {
    public:
        Driver(Model *model, State *state);
        virtual ~Driver();

#ifdef PARALLEL
        void wait_event(IOType to_type, cudaEvent_t *event);
#endif
        void schedule_from(IOType from_type);
        void schedule_to(IOType to_type);

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_calc_output();
        void stage_send_output();
        void stage_remaining();

        void step_all_states();
        void step_states(IOType layer_type);
        void step_weights();

        /* Cycles neuron states */
        //virtual void update_state(int start_index, int count) = 0;

        /* Updates weights for plastic neural connections.
         * TODO: implement.  This should use STDP variant Hebbian learning */
        void update_weights(Instruction *inst) { }

        /* Clears input of non-input neurons */
        void clear_input();
        /* Steps activation of a connection */
#ifdef PARALLEL
        void step_connection(Instruction *inst, cudaStream_t *stream);
#else
        void step_connection(Instruction *inst);
#endif

        State *state;
        std::vector<Instruction* > all_instructions;
        StreamCluster stream_clusters[IO_TYPE_SIZE];

#ifdef PARALLEL

        cudaStream_t io_stream;
        cudaStream_t kernel_stream;
        cudaStream_t *curr_stream;

        cudaEvent_t
            *input_event,
            *clear_event,
            *output_calc_event,
            *output_event;
#else
#endif
};

/* Instantiates a driver based on the driver_string in the given model */
Driver* build_driver(Model* model);

#endif
