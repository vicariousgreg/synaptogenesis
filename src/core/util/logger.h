#ifndef logger_h
#define logger_h

#include <string>

#define LOG_ERROR(ARG) Logger::log_error(ARG, __FILE__, __LINE__)
#define LOG_WARNING(ARG) Logger::log_warning(ARG, __FILE__, __LINE__)

#ifdef DEBUG
#define LOG_DEBUG(ARG) Logger::log_debug(ARG, __FILE__, __LINE__)
#else
#define LOG_DEBUG(ARG)
#endif

class DebugError : public std::exception { };

class Logger {
    public:
        static void log_warning(std::string error, const char* file, int line);
        static void log_error(std::string error, const char* file, int line);
        static void log_debug(std::string error, const char* file, int line);

        static bool suppress_output;
        static bool warnings;
        static bool debug;
};

#endif
