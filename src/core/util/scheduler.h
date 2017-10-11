#ifndef scheduler_h
#define scheduler_h

#include <functional>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <queue>
#include <set>
#include <vector>
#include <map>

#include "util/stream.h"
#include "util/event.h"

class Scheduler {
    public:
        void add(Stream *stream);

        void enqueue_wait(Stream *stream, Event *event);
        void enqueue_record(Stream *stream, Event *event);
        void enqueue_compute(Stream *stream, std::function<void()> f);

        void synchronize(Event* event);
        void start(int size);
        void shutdown();

        static Scheduler *get_instance();

    protected:
        static Scheduler *instance;

        Scheduler()
            : index(0), main_blocked_on(nullptr), running(false), dormant(false) { }
        virtual ~Scheduler() { shutdown(); }

        int index;
        std::vector<Stream*> streams;

        // Mutexed variables
        std::map<Stream*, bool> available_streams;
        std::map<Event*, std::vector<Stream*>> frozen_streams;
        std::map<Stream*, std::queue<std::function<void()>>> queues;
        std::set<Event*> enq;
        Event* main_blocked_on;
        bool dormant;

        // Thread variables
        std::vector<std::thread> threads;
        std::map<Stream*, std::mutex> stream_mutex;
        std::mutex event_mutex;
        std::mutex main_mutex;
        std::mutex dormant_mutex;

        std::condition_variable dormant_cv;
        std::condition_variable main_cv;
        bool running;

        void wait(Event* event, Stream* stream);
        void record(Event* event, Stream* stream);
        void compute(std::function<void()> f, Stream* stream);

        bool enqueued(Event *event);
        void enqueue(Event *event);
        void enqueue_and_thaw(Event *event);
        void dequeue(Event *event);
        void push(Stream *stream, std::function<void()> f);
        void pop(Stream *stream);
        bool freeze(Stream *stream, Event *event);
        void thaw(Event *event);
        void mark_available(Stream *stream);
        bool try_take(Stream *stream);
        void release(Stream *stream);
        void notify_dormant();
        void notify_main(Event *event);

        void worker_loop();
};

#endif
