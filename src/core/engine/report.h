#ifndef report_h
#define report_h

class Report {
    public:
        Report(int iterations, float total_time)
            : iterations(iterations),
              total_time(total_time),
              average_time(total_time / iterations) { }

        const int iterations;
        const float total_time;
        const float average_time;
};

#endif
