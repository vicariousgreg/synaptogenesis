#ifndef context_h
#define context_h

#include <vector>

#include "engine/report.h"

class Network;
class State;
class Environment;

class Context {
    public:
        Context(Network *network = nullptr,
                Environment *env = nullptr,
                State *st = nullptr);

        void free();

        void set_network(Network *net);
        void set_environment(Environment *env);
        void set_state(State *st);

        Network *get_network();
        Environment *get_environment();
        State *get_state();

        void add_report(Report* report) { reports.push_back(report); }
        const std::vector<Report*>& get_reports() { return reports; }
        Report* get_last_report() { return reports[reports.size()-1]; }

    private:
        Network* network;
        Environment* environment;
        State* state;
        std::vector<Report*> reports;
};

#endif
