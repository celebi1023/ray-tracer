ray: *.cu *.cuh
	mkdir -p build
	nvcc -o build/main.o -dc main.cu
	nvcc -o build/light.o -dc light.cu
	nvcc -o ray build/main.o build/light.o

clean:
	rm -rf build ray