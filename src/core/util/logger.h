#ifndef logger_h
#define logger_h

#include <string>

#define LOG_ERROR Logger::log_error
#define LOG_WARNING Logger::log_warning

#ifdef DEBUG
#define LOG_DEBUG(ARG) Logger::log_debug(ARG)
#else
#define LOG_DEBUG(ARG)
#endif

class DebugError : public std::exception { };

class Logger {
    public:
        static void log_warning(std::string error);
        static void log_error(std::string error);
        static void log_debug(std::string error);

        static bool suppress_output;
        static bool warnings;
        static bool debug;
};

#endif
