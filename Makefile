.PHONY: all clean plugins directories

CFLAGS := -Wall -Werror -O3 -g -fPIC

PLUGIN_DEPENDENCIES := src/plugin_interface.h obj/plugin_utilities.o \
	obj/plugin_hip_utilities.o

all: directories plugins bin/runner bin/hip_host_utilities.so

plugins: bin/mandelbrot.so bin/counter_spin.so

directories:
	mkdir -p obj
	mkdir -p bin

obj/cJSON.o: src/third_party/cJSON.c src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/cJSON.o src/third_party/cJSON.c

obj/runner.o: src/runner.c src/plugin_interface.h
	gcc -c $(CFLAGS) -o obj/runner.o src/runner.c

obj/parse_config.o: src/parse_config.c src/parse_config.h
	gcc -c $(CFLAGS) -o obj/parse_config.o src/parse_config.c

obj/barrier_wait.o: src/barrier_wait.c src/barrier_wait.h
	gcc -c $(CFLAGS) -o obj/barrier_wait.o src/barrier_wait.c

obj/plugin_hip_utilities.o: src/plugin_hip_utilities.cpp \
	src/plugin_hip_utilities.h
	hipcc -c $(CFLAGS) -o obj/plugin_hip_utilities.o \
		src/plugin_hip_utilities.cpp

obj/plugin_utilities.o: src/plugin_utilities.c src/plugin_utilities.h
	gcc -c $(CFLAGS) -o obj/plugin_utilities.o src/plugin_utilities.c

bin/mandelbrot.so: src/mandelbrot.cpp $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/mandelbrot.so src/mandelbrot.cpp \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o

bin/counter_spin.so: src/counter_spin.cpp $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/counter_spin.so src/counter_spin.cpp \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o

bin/hip_host_utilities.so: src/hip_host_utilities.cpp src/plugin_interface.h
	hipcc --shared $(CFLAGS) -o bin/hip_host_utilities.so \
		src/hip_host_utilities.cpp

bin/runner: obj/runner.o obj/cJSON.o obj/parse_config.o obj/barrier_wait.o
	gcc $(CFLAGS) -o bin/runner obj/runner.o obj/cJSON.o obj/parse_config.o \
		obj/barrier_wait.o obj/plugin_utilities.o -ldl -lpthread

clean:
	rm -r obj/
	rm -r bin/
