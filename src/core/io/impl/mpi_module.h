#ifdef __MPI__

#ifndef mpi_module_h
#define mpi_module_h

#include <vector>
#include "io/module.h"
#include "mpi_wrap.h"

class MPIModule : public Module {
    public:
        MPIModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

        virtual void report(Report* report);

    protected:
        int mpi_rank;

        // Unique tags for each layer
        std::map<int, Layer*> layer_tags;

        // Input/output layers for each source/destination
        std::map<int, std::vector<int>> input_tags;
        std::map<int, std::vector<int>> output_tags;

        // Input/output buffers
        std::map<int, Pointer<float>> output_buffers;
        std::map<int, Pointer<float>> input_buffers;

        // Indices, offsets, and sizes
        // Buffers are reused when possible
        std::map<int, int> buffer_indices;
        std::map<int, int> buffer_offsets;
        std::map<int, int> buffer_sizes;

        // MPI requests
        std::map<int, void*> send_requests;
        int recv_requests_array;
        std::map<int, int> recv_requests;
        std::map<int, int> recv_requests_inv;

    MODULE_MEMBERS
};

class MPILockstepModule : public MPIModule {
    public:
        MPILockstepModule(LayerList layers, ModuleConfig *config);

    MODULE_MEMBERS
};

#endif

#endif
