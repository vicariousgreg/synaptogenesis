#ifndef engine_h
#define engine_h

#include <vector>
#include <map>

class Layer;
class State;
class Environment;
class Cluster;
class ClusterNode;
class InterDeviceTransferInstruction;

class Engine {
    public:
        Engine(State *state, Environment *environment);
        virtual ~Engine();

        void set_learning_flag(bool status) { learning_flag = status; }

        // Main hooks
        void stage_clear();
        void stage_input();
        void stage_output();
        void stage_calc();

    protected:
        State *state;
        Environment *environment;
        std::vector<Cluster*> clusters;
        std::vector<InterDeviceTransferInstruction*> inter_device_transfers;
        std::map<Layer*, ClusterNode*> cluster_nodes;
        bool learning_flag;
};

#endif
