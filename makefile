all:
	make serial # parallel

parallel:
	nvcc -w -Wno-deprecated-gpu-targets -x cu *.cpp -DPARALLEL -o test

serial: serial_main.o implementations/serial_izhikevich_state.o implementations/serial_izhikevich_driver.o implementations/serial_rate_encoding_state.o implementations/serial_rate_encoding_driver.o framework/serial_driver.o framework/serial_model.o framework/serial_state.o framework/serial_input.o implementations/serial_random_input.o implementations/serial_image_input.o framework/serial_output.o implementations/serial_float_print_output.o implementations/serial_spike_print_output.o
	g++ -w -pthread serial_main.o framework/serial*.o implementations/serial*.o -o test

serial_main.o: main.cpp
	g++ -c -w main.cpp -o serial_main.o

implementations/serial_izhikevich_driver.o: implementations/izhikevich_driver.cpp implementations/izhikevich_driver.h
	g++ -c -w implementations/izhikevich_driver.cpp -o implementations/serial_izhikevich_driver.o

implementations/serial_izhikevich_state.o: implementations/izhikevich_state.cpp implementations/izhikevich_state.h
	g++ -c -w implementations/izhikevich_state.cpp -o implementations/serial_izhikevich_state.o

implementations/serial_rate_encoding_driver.o: implementations/rate_encoding_driver.cpp implementations/rate_encoding_driver.h
	g++ -c -w implementations/rate_encoding_driver.cpp -o implementations/serial_rate_encoding_driver.o

implementations/serial_rate_encoding_state.o: implementations/rate_encoding_state.cpp implementations/rate_encoding_state.h
	g++ -c -w implementations/rate_encoding_state.cpp -o implementations/serial_rate_encoding_state.o

framework/serial_driver.o: framework/driver.cpp framework/driver.h
	g++ -c -w framework/driver.cpp -o framework/serial_driver.o

framework/serial_input.o: framework/input.cpp framework/input.h
	g++ -c -w framework/input.cpp -o framework/serial_input.o

implementations/serial_random_input.o: implementations/random_input.cpp implementations/random_input.h
	g++ -c -w implementations/random_input.cpp -o implementations/serial_random_input.o

implementations/serial_image_input.o: implementations/image_input.cpp implementations/image_input.h
	g++ -c -w -pthread implementations/image_input.cpp -o implementations/serial_image_input.o

framework/serial_output.o: framework/output.cpp framework/output.h
	g++ -c -w framework/output.cpp -o framework/serial_output.o

implementations/serial_spike_print_output.o: implementations/spike_print_output.cpp implementations/spike_print_output.h
	g++ -c -w implementations/spike_print_output.cpp -o implementations/serial_spike_print_output.o

implementations/serial_float_print_output.o: implementations/float_print_output.cpp implementations/float_print_output.h
	g++ -c -w -pthread implementations/float_print_output.cpp -o implementations/serial_image_output.o

framework/serial_model.o: framework/model.cpp framework/model.h
	g++ -c -w framework/model.cpp -o framework/serial_model.o

framework/serial_state.o: framework/state.cpp framework/state.h
	g++ -c -w framework/state.cpp -o framework/serial_state.o


clean:
	rm *.o implementations/*.o  framework/*.o

