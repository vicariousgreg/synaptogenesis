#ifndef heatmap_h
#define heatmap_h

#include "visualizer.h"

class Heatmap : public Visualizer {
    public:
        static Heatmap *get_instance(bool init);

        virtual ~Heatmap();

        virtual bool add_output_layer(Layer *layer, std::string params);
        virtual void update(Environment *environment);
        virtual std::string get_name() { return Heatmap::name; }

    private:
        Heatmap();
        static std::string name;

        int iterations;
        std::map<int, float*> spike_count_map;
};

#endif
