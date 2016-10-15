#ifndef buffer_h
#define buffer_h

class Buffer {
    public:
        Buffer(int num_neurons, int output_size);
        ~Buffer() {
            free(this->input);
            free(this->output);
        }

        void set_input(int offset, int size, float* source);

        void set_output(int offset, int size, void* source);

        float* get_input();
        void* get_output();

    private:
        int output_size;
        int num_neurons;
        float *input;
        void *output;
};

#endif
