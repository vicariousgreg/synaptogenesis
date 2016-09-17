all:
	make parallel

parallel:
	nvcc -x cu *.cpp -Dparallel -o test

serial:
	g++ -w *.cpp *.h -o test

clean:
	rm *.gch

