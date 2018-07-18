#ifdef __GUI__

#ifndef saccade_module_h
#define saccade_module_h

#include "io/module.h"

class SaccadeWindow;

class SaccadeModule : public Module {
    public:
        SaccadeModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

        virtual void report(Report* report);

        void init();
        void update(Buffer *buffer);

        void log_correct(bool correct) { correct_log.push_back(correct); }
        void log_time(int time) { time_log.push_back(time); }

    private:
        std::map<Layer*, bool> central;
        SaccadeWindow *window;

        std::vector<bool> correct_log;
        std::vector<int> time_log;

    MODULE_MEMBERS
};

#endif

#endif
