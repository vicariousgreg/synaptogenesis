#ifndef error_manager_h
#define error_manager_h

#include <string>

#define LOG_ERROR ErrorManager::log_error
#define LOG_WARNING ErrorManager::log_warning

#ifdef DEBUG
#define LOG_DEBUG(ARG) ErrorManager::log_debug(ARG)
#else
#define LOG_DEBUG(ARG)
#endif

class DebugError : public std::exception { };

class ErrorManager {
    public:
        static void log_warning(std::string error);
        static void log_error(std::string error);
        static void log_debug(std::string error);

        static bool suppress_output;
        static bool warnings;
        static bool debug;
};

#endif
