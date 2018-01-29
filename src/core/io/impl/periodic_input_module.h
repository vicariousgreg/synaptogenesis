#ifndef periodic_input_module_h
#define periodic_input_module_h

#include "io/module.h"
#include "util/pointer.h"

class PeriodicInputModule : public Module {
    public:
        PeriodicInputModule(LayerList layers, ModuleConfig *config);
        virtual ~PeriodicInputModule();

        void feed_input_impl(Buffer *buffer);
        void cycle_impl();

    protected:
        virtual void update() = 0;

        Pointer<float> values;

        float value;
        float min_value;
        float max_value;
        float fraction;
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

class GaussianRandomInputModule : public PeriodicInputModule {
    public:
        GaussianRandomInputModule(LayerList layers, ModuleConfig *config);
        virtual ~GaussianRandomInputModule();

    protected:
        virtual void update();

        Pointer<float> gaussians;
        int rows, columns;
        int gauss_rows, gauss_columns;

    MODULE_MEMBERS
};

#endif
