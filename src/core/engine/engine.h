#ifndef engine_h
#define engine_h

#include <vector>
#include <map>
#include <thread>
#include <mutex>

#include "engine/context.h"
#include "io/buffer.h"
#include "io/module.h"
#include "util/constants.h"

class Layer;
class Context;
class Report;
class State;
class Cluster;
class ClusterNode;
class InterDeviceTransferInstruction;

enum Thread_ID {
    NETWORK_THREAD,
    ENVIRONMENT_THREAD
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
        Engine(Context context);
        virtual ~Engine();

        // Clears the buffer, modules, cluster, and resources
        void clear();

        // Rebuilds the engine
        // Necessary after changes to network or environment
        void rebuild();

        // Run the engine
        Report* run(PropertyConfig args=PropertyConfig());

        void interrupt();

        Buffer* get_buffer() { return buffer; }
        IOTypeMask get_io_type(Layer *layer) { return io_types[layer]; }
        bool is_input(Layer *layer) { return get_io_type(layer) & INPUT; }
        bool is_output(Layer *layer) { return get_io_type(layer) & OUTPUT; }
        bool is_expected(Layer *layer) { return get_io_type(layer) & EXPECTED; }

        size_t get_buffer_bytes() const;

    protected:
        Context context;
        bool running;

        void build_environment();
        void build_clusters();

        // Cluster data
        std::vector<Cluster*> clusters;
        std::map<Layer*, ClusterNode*> cluster_nodes;
        std::vector<InterDeviceTransferInstruction*> inter_device_transfers;

        // Environment data
        Buffer* buffer;
        ModuleList modules;
        std::map<Layer*, IOTypeMask> io_types;
        bool suppress_output;

        // Running data
        Lock sensory_lock;
        Lock motor_lock;
        Timer run_timer;
        Timer iteration_timer;
        float refresh_rate, time_limit;
        bool calc_rate;
        int environment_rate;
        bool learning_flag;

        size_t iterations;
        bool verbose;
        Report *report;

        // Thread loops
        void network_loop();
        void environment_loop();
};

#endif
