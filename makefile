all:
	make parallel

parallel:
	nvcc -x cu *.cpp -DPARLLEL -o test

serial:
	g++ -w *.cpp *.h -o test

clean:
	rm *.gch

