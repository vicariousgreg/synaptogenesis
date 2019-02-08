#ifndef engine_h
#define engine_h

#include <vector>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "context.h"
#include "io/buffer.h"
#include "io/module.h"
#include "util/constants.h"
#include "util/timer.h"

class Layer;
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
        Thread_ID get_owner()               volatile { return owner; }
        void set_owner(Thread_ID new_owner) volatile { owner = new_owner; }

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
        // Volatile owner to ensure thread-safety with optimizations
        volatile Thread_ID owner;
        std::mutex mutex;
};

class Engine {
    public:
        Engine(Context context);
        virtual ~Engine();

        // Clears the buffer, modules, cluster, and resources
        void clear();

        // Rebuilds the engine
        // Necessary after changes to network or environment
        void rebuild(PropertyConfig args=PropertyConfig());

        // Run the engine
        Report* run(PropertyConfig args=PropertyConfig());

        Buffer* get_buffer() { return buffer; }
        IOTypeMask get_io_type(Layer *layer) { return io_types[layer]; }
        KeySet get_input_keys(Layer* layer) { return input_keys[layer]; }
        KeySet get_output_keys(Layer* layer) { return output_keys[layer]; }
        bool is_input(Layer *layer) { return get_io_type(layer) & INPUT; }
        bool is_output(Layer *layer) { return get_io_type(layer) & OUTPUT; }

        size_t get_buffer_bytes() const;

        // Interrupts all active engines
        static void interrupt();
        static void interrupt_async();

    protected:
        Context context;
        bool network_running;
        bool environment_running;

        void build_environment(PropertyConfig args);
        void build_clusters(PropertyConfig args);

        // Cluster data
        std::vector<Cluster*> clusters;
        std::map<Layer*, ClusterNode*> cluster_nodes;
        std::vector<InterDeviceTransferInstruction*> inter_device_transfers;

        // Environment data
        Buffer* buffer;
        ModuleList modules;
        std::map<Layer*, IOTypeMask> io_types;
        bool suppress_output;
        LayerKeyMap input_keys;
        LayerKeyMap output_keys;

        // Running data
        Lock sensory_lock;
        Lock motor_lock;
        Lock term_lock;
        Timer run_timer;
        Timer iteration_timer;
        float refresh_rate, time_limit;
        int environment_rate;
        bool learning_flag;

        size_t iterations;
        bool verbose;
        Report *report;

        // Thread loops
        bool multithreaded;
        bool killed;
        void single_thread_loop();
        void network_loop();
        void environment_loop();

        // Static infrastructure for interruption
        static std::mutex interrupt_lock;
        static bool interrupt_signaled;
        static bool running;
};

#endif
