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

    private:
        // Source and destination ranks
        std::map<Layer*, int> sources;
        std::map<Layer*, std::vector<int>> destinations;

        // Unique tags for each layer
        std::map<Layer*, int> tags;

        // Local buffers and MPI requests for sends
        std::map<Layer*, Pointer<float>> local_buffers;
        std::map<Layer*, std::vector<int>> requests;

    MODULE_MEMBERS
};

#endif

#endif
