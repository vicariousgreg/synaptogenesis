#ifndef visualizer_h
#define visualizer_h

#include "model/layer.h"
#include "layer_info.h"

class GUI;
class Environment;

class Visualizer {
    public:
        Visualizer(Environment *environment);
        virtual ~Visualizer();

        void add_layer(Layer *layer, bool input, bool output);
        void launch();
        void update();

    private:
        Environment *environment;
        GUI *gui;
};

#endif
