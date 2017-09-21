#ifndef context_h
#define context_h

#include <vector>

#include "engine/report.h"

class Network;
class State;
class Environment;

class Context {
    public:
        Context(Network *network,
                Environment *env = nullptr,
                State *st = nullptr);

        virtual ~Context();

        Network *get_network() { return network; }
        Environment *get_environment() { return environment; }
        State *get_state() { return state; }

        void add_report(Report report) { reports.push_back(report); }
        std::vector<Report> get_reports() { return reports; }
        Report get_last_report() { return reports[reports.size()-1]; }

    private:
        Network* network;
        Environment* environment;
        State* state;
        std::vector<Report> reports;
};

#endif
