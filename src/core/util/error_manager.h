#ifndef error_manager_h
#define error_manager_h

#include <string>

class ErrorManager {
    public:
        static ErrorManager *get_instance();
        void suppress_warnings() { warnings = false; }

        void log_warning(std::string error);
        void log_error(std::string error);

    private:
        static ErrorManager *instance;
        ErrorManager() : warnings(true) { }

        bool warnings;
};

#endif
