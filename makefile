all:
	make serial # parallel

parallel:
	nvcc -w -Wno-deprecated-gpu-targets -x cu *.cpp -DPARALLEL -o test

serial: build/serial/main.o build/serial/implementation/izhikevich_state.o build/serial/implementation/izhikevich_driver.o build/serial/implementation/rate_encoding_state.o build/serial/implementation/rate_encoding_driver.o build/serial/framework/driver.o build/serial/framework/model.o build/serial/framework/state.o build/serial/framework/input.o build/serial/implementation/random_input.o build/serial/implementation/image_input.o build/serial/framework/output.o build/serial/implementation/float_print_output.o build/serial/implementation/spike_print_output.o
	g++ -w -pthread build/serial/main.o build/serial/framework/*.o build/serial/implementation/*.o -o test

build/serial/main.o: src/main.cpp
	g++ -c -w src/main.cpp -o build/serial/main.o

build/serial/implementation/izhikevich_driver.o: src/implementation/izhikevich_driver.cpp src/implementation/izhikevich_driver.h
	g++ -c -w src/implementation/izhikevich_driver.cpp -o build/serial/implementation/izhikevich_driver.o

build/serial/implementation/izhikevich_state.o: src/implementation/izhikevich_state.cpp src/implementation/izhikevich_state.h
	g++ -c -w src/implementation/izhikevich_state.cpp -o build/serial/implementation/izhikevich_state.o

build/serial/implementation/rate_encoding_driver.o: src/implementation/rate_encoding_driver.cpp src/implementation/rate_encoding_driver.h
	g++ -c -w src/implementation/rate_encoding_driver.cpp -o build/serial/implementation/rate_encoding_driver.o

build/serial/implementation/rate_encoding_state.o: src/implementation/rate_encoding_state.cpp src/implementation/rate_encoding_state.h
	g++ -c -w src/implementation/rate_encoding_state.cpp -o build/serial/implementation/rate_encoding_state.o

build/serial/framework/driver.o: src/framework/driver.cpp src/framework/driver.h
	g++ -c -w src/framework/driver.cpp -o build/serial/framework/driver.o

build/serial/framework/input.o: src/framework/input.cpp src/framework/input.h
	g++ -c -w src/framework/input.cpp -o build/serial/framework/input.o

build/serial/implementation/random_input.o: src/implementation/random_input.cpp src/implementation/random_input.h
	g++ -c -w src/implementation/random_input.cpp -o build/serial/implementation/random_input.o

build/serial/implementation/image_input.o: src/implementation/image_input.cpp src/implementation/image_input.h
	g++ -c -w -pthread src/implementation/image_input.cpp -o build/serial/implementation/image_input.o

build/serial/framework/output.o: src/framework/output.cpp src/framework/output.h
	g++ -c -w src/framework/output.cpp -o build/serial/framework/output.o

build/serial/implementation/spike_print_output.o: src/implementation/spike_print_output.cpp src/implementation/spike_print_output.h
	g++ -c -w src/implementation/spike_print_output.cpp -o build/serial/implementation/spike_print_output.o

build/serial/implementation/float_print_output.o: src/implementation/float_print_output.cpp src/implementation/float_print_output.h
	g++ -c -w -pthread src/implementation/float_print_output.cpp -o build/serial/implementation/image_output.o

build/serial/framework/model.o: src/framework/model.cpp src/framework/model.h
	g++ -c -w src/framework/model.cpp -o build/serial/framework/model.o

build/serial/framework/state.o: src/framework/state.cpp src/framework/state.h
	g++ -c -w src/framework/state.cpp -o build/serial/framework/state.o


clean:
	find build/serial/*.o build/serial/implementation/*.o build/serial/framework/*.o | xargs rm

