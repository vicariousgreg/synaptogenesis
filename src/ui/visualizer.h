#ifndef visualizer_h
#define visualizer_h

#include <map>

#include "layer_info.h"

class GUI;
class VisualizerWindow;
class Layer;
class Environment;

class Visualizer {
    public:
        static Visualizer *get_instance(bool init);
        static void delete_instance();

        virtual ~Visualizer();

        void add_input_layer(Layer *layer);
        void add_output_layer(Layer *layer);
        void launch();
        void update(Environment *environment);

    private:
        static Visualizer *instance;
        Visualizer();

        GUI *gui;
        VisualizerWindow *window;
        std::map<Layer*, LayerInfo*> layer_map;
};

#endif
