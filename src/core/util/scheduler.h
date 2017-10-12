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
        static Scheduler *get_instance();
        virtual ~Scheduler();

        /* Launches a thread pool
         *
         * If size is 0, the scheduler will immediately run operations
         * Otherwise, they will be performed by worker threads */
        void start(unsigned int size);

        /* Shuts down the thread pool if it is running */
        void shutdown();

        /* Enqueue operations on the provided stream */
        void enqueue_wait(Stream *stream, Event *event);
        void enqueue_record(Stream *stream, Event *event);
        void enqueue_compute(Stream *stream, std::function<void()> f);

        /* Block the caller until the event is recorded */
        void synchronize(Event* event);

    protected:
        static Scheduler *instance;

        Scheduler()
            : index(0),
              client_blocked_on(nullptr),
              pool_running(false),
              dormant(false),
              single_thread(true) { }

        /* Indicates what a worker should do after performing an operation */
        typedef enum {
            CONTINUE,
            STOP_POP,
            STOP_NO_POP
        } QueueSignal;

        /* Worker functions */
        Stream* worker_get_stream();
        void worker_run_stream(Stream *stream);
        void worker_loop();
        QueueSignal wait(Event* event, Stream* stream);
        QueueSignal record(Event* event, Stream* stream);
        QueueSignal compute(std::function<void()> f, Stream* stream);

        /* Thread variables */
        std::vector<std::thread> threads;
        bool single_thread;

        /* Worker variables */
        std::mutex worker_mutex;
        std::vector<Stream*> streams;
        std::set<Event*> events;
        int index;
        bool pool_running;

        /* Stream variables
         * Stream queues contain operations to perform
         * Only one worker can be operating on a stream at once
         * Streams being operated on are temporarily marked as unavailable */
        std::map<Stream*, std::mutex> stream_mutex;
        std::map<Stream*, bool> available_streams;
        std::map<Stream*, std::queue<std::function<QueueSignal()>>> queues;
        void lock_stream(Stream *stream);
        void unlock_stream(Stream *stream);
        void push(Stream *stream, std::function<QueueSignal()> f);
        void pop(Stream *stream);
        void mark_available(Stream *stream);

        /* Event variables
         * When an event is queued up to be recorded, it becomes active
         * If a stream needs to wait for an event, it will be frozen if
         *   the event is active, and thawed when the event is recorded */
        std::mutex event_mutex;
        std::map<Event*, bool> active_events;
        std::map<Event*, std::vector<Stream*>> frozen_streams;
        bool active(Event *event);
        void activate(Event *event);
        void deactivate(Event *event);
        bool freeze(Stream *stream, Event *event);
        void thaw(Event *event);

        /* Client variables
         * Callers of synchronize(event) are blocked if the event is active
         * When the event is recorded, the client will be woken up */
        std::mutex client_mutex;
        std::condition_variable client_cv;
        Event* client_blocked_on;
        void block_client(Event *event);
        void notify_client(Event *event);

        /* Dormant variables
         * To avoid busy waiting, if there are no available streams
         *   (or available streams have empty queues), worker threads
         *   will become dormant
         * Threads can be awakened using notify_dormant(), which is done
         *   when new operations are enqueued, or the pool is shut down */
        std::mutex dormant_mutex;
        std::condition_variable dormant_cv;
        bool dormant;
        void notify_dormant();

    private:
        friend class Stream;
        friend class Event;

        /* Adds or removes a stream from the scheduler
         * Done automatically by Stream constructor/destructor */
        void add(Stream *stream);
        void remove(Stream *stream);

        /* Adds or removes an event from the scheduler
         * Done automatically by Event constructor/destructor */
        void add(Event *event);
        void remove(Event *event);
};

#endif
