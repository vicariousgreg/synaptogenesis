#ifndef buffer_h
#define buffer_h

class Buffer {
    public:
        Buffer(int num_neurons, int output_size);
        ~Buffer() {
// UNCOMMENT FOR PINNED MEMORY
//#ifdef PARALLEL
//            cudaFree(this->input);
//            cudaFree(this->output);
//#else
            free(this->input);
            free(this->output);
//#endif
        }

        void clear_input();
        void set_input(int offset, int size, float* source);

        float* get_input();
        void* get_output();

        int get_output_size() { return output_size; }

    private:
        int output_size;
        int num_neurons;
        float *input;
        void *output;
};

#endif
