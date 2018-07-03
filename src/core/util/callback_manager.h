#ifndef callback_manager_h
#define callback_manager_h

#include <string>
#include <map>

class CallbackManager {
    public:
        static CallbackManager *get_instance();

        void add_io_callback(std::string name,
            void (*addr)(int, int, void*));

        void add_weight_callback(std::string name,
            void (*addr)(int, int, void*));

        void add_indices_weight_callback(std::string name,
            void (*addr)(int, int, void*, void*, void*, void*));

        void add_distance_weight_callback(std::string name,
            void (*addr)(int, int, void*, void*));

        void add_delay_weight_callback(std::string name,
            void (*addr)(int, int, void*, void*));

        void (*get_io_callback(std::string name))(int, int, void*);

        void (*get_weight_callback(std::string name))
            (int, int, void*);

        void (*get_indices_weight_callback(std::string name))
            (int, int, void*, void*, void*, void*);

        void (*get_distance_weight_callback(std::string name))
            (int, int, void*, void*);

        void (*get_delay_weight_callback(std::string name))
            (int, int, void*, void*);

    private:
        static CallbackManager *instance;
        CallbackManager() { }

        std::map<std::string, void (*)(int, int, void*)> io_callbacks;
        std::map<std::string, void (*)(int, int, void*)>
            weight_callbacks;
        std::map<std::string, void (*)(int, int, void*, void*, void*, void*)>
            indices_weight_callbacks;
        std::map<std::string, void (*)(int, int, void*, void*)>
            distance_weight_callbacks;
        std::map<std::string, void (*)(int, int, void*, void*)>
            delay_weight_callbacks;
};

#endif
