#ifndef engine_h
#define engine_h

#include <vector>
#include <map>
#include <thread>
#include <mutex>

#include "io/buffer.h"
#include "io/module.h"
#include "util/constants.h"

class Layer;
class Context;
class State;
class Cluster;
class ClusterNode;
class InterDeviceTransferInstruction;

enum Thread_ID {
    NETWORK,
    ENVIRONMENT
};

class Lock {
    public:
        void set_owner(Thread_ID new_owner)
            { owner = new_owner; }

        void wait(Thread_ID me) {
            while (true) {
                mutex.lock();
                if (owner == me) return;
                else mutex.unlock();
                std::this_thread::yield();
            }
        }

        void pass(Thread_ID new_owner) {
            owner = new_owner;
            mutex.unlock();
        }

    private:
        std::mutex mutex;
        Thread_ID owner;
};

class Engine {
    public:
        Engine(Context *context, bool suppress_output=false);
        virtual ~Engine();

        // Rebuilds the engine
        // Necessary after changes to network or environment
        void rebuild();

        // Run the engine
        Context* run(int iterations, bool verbose);

        void set_learning_flag(bool status) { learning_flag = status; }
        void set_suppress_output(bool status) { suppress_output = status; }
        void set_calc_rate(bool status) { calc_rate = status; }
        void set_environment_rate(int rate) { environment_rate = rate; }
        void set_refresh_rate(float rate) {
            refresh_rate = rate;
            time_limit = 1.0 / refresh_rate;
        }

        Buffer* get_buffer() { return buffer; }
        IOTypeMask get_io_type(Layer *layer) { return io_types[layer]; }
        bool is_input(Layer *layer) { return get_io_type(layer) & INPUT; }
        bool is_output(Layer *layer) { return get_io_type(layer) & OUTPUT; }
        bool is_expected(Layer *layer) { return get_io_type(layer) & EXPECTED; }

    protected:
        Context *context;

        void build_environment();
        void build_clusters();

        // Cluster data
        std::vector<Cluster*> clusters;
        std::map<Layer*, ClusterNode*> cluster_nodes;
        std::vector<InterDeviceTransferInstruction*> inter_device_transfers;
        bool suppress_output;

        // Environment data
        std::map<Layer*, IOTypeMask> io_types;
        ModuleList modules;
        Buffer* buffer;

        // Running data
        Lock sensory_lock;
        Lock motor_lock;
        Timer run_timer;
        Timer iteration_timer;
        float refresh_rate, time_limit;
        bool calc_rate;
        int environment_rate;
        bool learning_flag;

        // Thread loops
        void network_loop(int iterations, bool verbose);
        void environment_loop(int iterations, bool verbose);
};

#endif
