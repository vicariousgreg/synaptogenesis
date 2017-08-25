#ifndef error_manager_h
#define error_manager_h

#include <string>

class DebugError : public std::exception { };

class ErrorManager {
    public:
        static ErrorManager *get_instance();
        void set_warnings(bool warnings) { this->warnings = warnings; }
        void set_debug(bool debug) { this->debug = debug; }
        void set_suppress_output(bool supp) { this->suppress_output = supp; }

        void log_warning(std::string error);
        void log_error(std::string error);
        void log_debug(std::string error);

    private:
        static ErrorManager *instance;
        ErrorManager() : warnings(true), debug(false) { }

        bool suppress_output;
        bool warnings;
        bool debug;
};

#endif
