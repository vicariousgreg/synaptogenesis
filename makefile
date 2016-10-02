all:
	make serial # parallel

parallel:
	nvcc -w -Wno-deprecated-gpu-targets -x cu *.cpp -DPARALLEL -o test

serial: serial_main.o implementation/serial_izhikevich_state.o implementation/serial_izhikevich_driver.o implementation/serial_rate_encoding_state.o implementation/serial_rate_encoding_driver.o framework/serial_driver.o framework/serial_model.o framework/serial_state.o framework/serial_input.o implementation/serial_random_input.o implementation/serial_image_input.o framework/serial_output.o implementation/serial_float_print_output.o implementation/serial_spike_print_output.o
	g++ -w -pthread serial_main.o framework/serial*.o implementation/serial*.o -o test

serial_main.o: main.cpp
	g++ -c -w main.cpp -o serial_main.o

implementation/serial_izhikevich_driver.o: implementation/izhikevich_driver.cpp implementation/izhikevich_driver.h
	g++ -c -w implementation/izhikevich_driver.cpp -o implementation/serial_izhikevich_driver.o

implementation/serial_izhikevich_state.o: implementation/izhikevich_state.cpp implementation/izhikevich_state.h
	g++ -c -w implementation/izhikevich_state.cpp -o implementation/serial_izhikevich_state.o

implementation/serial_rate_encoding_driver.o: implementation/rate_encoding_driver.cpp implementation/rate_encoding_driver.h
	g++ -c -w implementation/rate_encoding_driver.cpp -o implementation/serial_rate_encoding_driver.o

implementation/serial_rate_encoding_state.o: implementation/rate_encoding_state.cpp implementation/rate_encoding_state.h
	g++ -c -w implementation/rate_encoding_state.cpp -o implementation/serial_rate_encoding_state.o

framework/serial_driver.o: framework/driver.cpp framework/driver.h
	g++ -c -w framework/driver.cpp -o framework/serial_driver.o

framework/serial_input.o: framework/input.cpp framework/input.h
	g++ -c -w framework/input.cpp -o framework/serial_input.o

implementation/serial_random_input.o: implementation/random_input.cpp implementation/random_input.h
	g++ -c -w implementation/random_input.cpp -o implementation/serial_random_input.o

implementation/serial_image_input.o: implementation/image_input.cpp implementation/image_input.h
	g++ -c -w -pthread implementation/image_input.cpp -o implementation/serial_image_input.o

framework/serial_output.o: framework/output.cpp framework/output.h
	g++ -c -w framework/output.cpp -o framework/serial_output.o

implementation/serial_spike_print_output.o: implementation/spike_print_output.cpp implementation/spike_print_output.h
	g++ -c -w implementation/spike_print_output.cpp -o implementation/serial_spike_print_output.o

implementation/serial_float_print_output.o: implementation/float_print_output.cpp implementation/float_print_output.h
	g++ -c -w -pthread implementation/float_print_output.cpp -o implementation/serial_image_output.o

framework/serial_model.o: framework/model.cpp framework/model.h
	g++ -c -w framework/model.cpp -o framework/serial_model.o

framework/serial_state.o: framework/state.cpp framework/state.h
	g++ -c -w framework/state.cpp -o framework/serial_state.o


clean:
	rm *.o implementation/*.o  framework/*.o

