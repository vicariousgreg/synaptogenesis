#ifndef periodic_input_module_h
#define periodic_input_module_h

#include "io/module.h"
#include "util/pointer.h"

class PeriodicInputModule : public Module {
    public:
        PeriodicInputModule(LayerList layers, ModuleConfig *config);
        virtual ~PeriodicInputModule();

        void feed_input(Buffer *buffer);
        void cycle();

    protected:
        virtual void update() = 0;

        Pointer<float> values;

        int timesteps;
        int rate;
        int end;
        float value;
        float min_value;
        float max_value;
        float fraction;
        bool verbose;
        bool clear;
        bool random;
        bool dirty;
};

class BasicPeriodicInputModule : public PeriodicInputModule {
    public:
        BasicPeriodicInputModule(LayerList layers, ModuleConfig *config)
            : PeriodicInputModule(layers, config) { this->update(); }

    protected:
        virtual void update();

    MODULE_MEMBERS
};

class OneHotRandomInputModule : public PeriodicInputModule {
    public:
        OneHotRandomInputModule(LayerList layers, ModuleConfig *config)
            : PeriodicInputModule(layers, config) { this->update(); }

    protected:
        virtual void update();

    MODULE_MEMBERS
};

class OneHotCyclicInputModule : public PeriodicInputModule {
    public:
        OneHotCyclicInputModule(LayerList layers, ModuleConfig *config)
            : PeriodicInputModule(layers, config) { this->update(); }

    protected:
        virtual void update();

    MODULE_MEMBERS
};

#endif
