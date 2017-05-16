#ifndef auditory_cortex_h
#define auditory_cortex_h

#include <vector>
#include <string>

#include "model/model.h"
#include "model/structure.h"

class AuditoryCortex : public Structure {
    public:
        AuditoryCortex(Model *model, int spec_size, int spec_spread);

        void add_input(std::string layer, std::string input_name,
            std::string module_name, std::string module_params);

    protected:
        int spec_size;
        int spec_spread;
        int cortex_rows;
        int cortex_cols;

        void add_cortical_layer(std::string name, bool shifted,
            int size_fraction=1, int conn_fraction=1);
        void connect_one_way(std::string name1, std::string name2,
            int spread, float fraction, int delay=0, int stride=1);
        void connect_reentrant(std::string name1, std::string name2,
            int spread, float fraction, int delay=0, int stride=1);
        void connect_tether(std::string name1, std::string name2,
            int num_tethers, int tether_size, float fraction, int delay=0);
};

#endif
