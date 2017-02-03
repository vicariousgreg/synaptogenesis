#ifndef error_manager_h
#define error_manager_h

#include <string>

class ErrorManager {
    public:
        void log_warning(std::string error);
        void log_error(std::string error);
        static ErrorManager *get_instance();

    private:
        static ErrorManager *instance;
        ErrorManager() { }
};

#endif
