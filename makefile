all:
	make serial # parallel

parallel:
	nvcc -w -Wno-deprecated-gpu-targets -x cu *.cpp -DPARALLEL -o test

serial: serial_main.o serial_izhikevich_state.o serial_izhikevich_driver.o serial_rate_encoding_state.o serial_rate_encoding_driver.o serial_driver.o serial_model.o serial_state.o serial_input.o serial_random_input.o serial_image_input.o serial_output.o serial_float_print_output.o serial_spike_print_output.o
	g++ -w -pthread serial*.o -o test

serial_main.o: main.cpp
	g++ -c -w main.cpp -o serial_main.o

serial_izhikevich_driver.o: izhikevich_driver.cpp izhikevich_driver.h
	g++ -c -w izhikevich_driver.cpp -o serial_izhikevich_driver.o

serial_izhikevich_state.o: izhikevich_state.cpp izhikevich_state.h
	g++ -c -w izhikevich_state.cpp -o serial_izhikevich_state.o

serial_rate_encoding_driver.o: rate_encoding_driver.cpp rate_encoding_driver.h
	g++ -c -w rate_encoding_driver.cpp -o serial_rate_encoding_driver.o

serial_rate_encoding_state.o: rate_encoding_state.cpp rate_encoding_state.h
	g++ -c -w rate_encoding_state.cpp -o serial_rate_encoding_state.o

serial_driver.o: driver.cpp driver.h
	g++ -c -w driver.cpp -o serial_driver.o

serial_input.o: input.cpp input.h
	g++ -c -w input.cpp -o serial_input.o

serial_random_input.o: random_input.cpp random_input.h
	g++ -c -w random_input.cpp -o serial_random_input.o

serial_image_input.o: image_input.cpp image_input.h
	g++ -c -w -pthread image_input.cpp -o serial_image_input.o

serial_output.o: output.cpp output.h
	g++ -c -w output.cpp -o serial_output.o

serial_spike_print_output.o: spike_print_output.cpp spike_print_output.h
	g++ -c -w spike_print_output.cpp -o serial_spike_print_output.o

serial_float_print_output.o: float_print_output.cpp float_print_output.h
	g++ -c -w -pthread float_print_output.cpp -o serial_image_output.o

serial_model.o: model.cpp model.h
	g++ -c -w model.cpp -o serial_model.o

serial_state.o: state.cpp state.h
	g++ -c -w state.cpp -o serial_state.o


clean:
	rm *.o

