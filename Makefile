.PHONY: all clean plugins directories rodinia_plugins

CFLAGS := -Wall -Werror -O3 -g -fPIC

PLUGIN_DEPENDENCIES := src/plugin_interface.h obj/plugin_utilities.o \
	obj/plugin_hip_utilities.o

RODINIA_DIR := src/third_party/rodinia_plugins

all: directories plugins bin/runner bin/hip_host_utilities.so bin/list_devices \
	bin/test_clock

plugins: directories bin/mandelbrot.so bin/counter_spin.so bin/timer_spin.so \
	bin/random_walk.so bin/huge_kernels.so bin/memory_copy.so \
	bin/vector_add.so bin/dummy_streams.so

rodinia_plugins: directories obj/cJSON.o obj/plugin_utilities.o \
	obj/plugin_hip_utilities.o
	+cd $(RODINIA_DIR) && make

directories:
	mkdir -p obj
	mkdir -p bin

obj/cJSON.o: src/third_party/cJSON.c src/third_party/cJSON.h
	gcc -c $(CFLAGS) -o obj/cJSON.o src/third_party/cJSON.c

obj/runner.o: src/runner.c src/plugin_interface.h
	gcc -c $(CFLAGS) -o obj/runner.o src/runner.c

obj/parse_config.o: src/parse_config.c src/parse_config.h
	gcc -c $(CFLAGS) -o obj/parse_config.o src/parse_config.c

obj/multiprocess_sync.o: src/multiprocess_sync.c src/multiprocess_sync.h
	gcc -c $(CFLAGS) -o obj/multiprocess_sync.o src/multiprocess_sync.c

obj/plugin_hip_utilities.o: src/plugin_hip_utilities.cpp \
	src/plugin_hip_utilities.h
	hipcc -c $(CFLAGS) -o obj/plugin_hip_utilities.o \
		src/plugin_hip_utilities.cpp

obj/plugin_utilities.o: src/plugin_utilities.c src/plugin_utilities.h
	gcc -c $(CFLAGS) -o obj/plugin_utilities.o src/plugin_utilities.c

obj/mandelbrot.o: src/mandelbrot.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/mandelbrot.o src/mandelbrot.cpp

bin/mandelbrot.so: obj/mandelbrot.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/mandelbrot.so obj/mandelbrot.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o obj/cJSON.o

obj/counter_spin.o: src/counter_spin.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/counter_spin.o src/counter_spin.cpp

bin/counter_spin.so: obj/counter_spin.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/counter_spin.so obj/counter_spin.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o

obj/timer_spin.o: src/timer_spin.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/timer_spin.o src/timer_spin.cpp

bin/timer_spin.so: obj/timer_spin.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/timer_spin.so obj/timer_spin.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o

obj/random_walk.o: src/random_walk.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/random_walk.o src/random_walk.cpp

bin/random_walk.so: obj/random_walk.o obj/cJSON.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/random_walk.so obj/random_walk.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o obj/cJSON.o

obj/memory_copy.o: src/memory_copy.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/memory_copy.o src/memory_copy.cpp

bin/memory_copy.so: obj/memory_copy.o obj/cJSON.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/memory_copy.so obj/memory_copy.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o obj/cJSON.o

obj/huge_kernels.o: src/huge_kernels.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/huge_kernels.o src/huge_kernels.cpp

bin/huge_kernels.so: obj/huge_kernels.o obj/cJSON.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/huge_kernels.so obj/huge_kernels.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o obj/cJSON.o

obj/vector_add.o: src/vector_add.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/vector_add.o src/vector_add.cpp

bin/vector_add.so: obj/vector_add.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/vector_add.so obj/vector_add.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o obj/cJSON.o

obj/dummy_streams.o: src/dummy_streams.cpp $(PLUGIN_DEPENDENCIES)
	hipcc -c $(CFLAGS) -o obj/dummy_streams.o src/dummy_streams.cpp

bin/dummy_streams.so: obj/dummy_streams.o $(PLUGIN_DEPENDENCIES)
	hipcc --shared $(CFLAGS) -o bin/dummy_streams.so obj/dummy_streams.o \
		obj/plugin_utilities.o obj/plugin_hip_utilities.o obj/cJSON.o

bin/hip_host_utilities.so: src/hip_host_utilities.cpp src/plugin_interface.h
	hipcc --shared $(CFLAGS) -o bin/hip_host_utilities.so \
		src/hip_host_utilities.cpp

bin/runner: obj/runner.o obj/cJSON.o obj/parse_config.o obj/multiprocess_sync.o
	gcc $(CFLAGS) -o bin/runner obj/runner.o obj/cJSON.o obj/parse_config.o \
		obj/multiprocess_sync.o obj/plugin_utilities.o -ldl -lpthread

bin/list_devices: src/list_devices.cpp
	hipcc -o bin/list_devices $(CFLAGS) src/list_devices.cpp

bin/test_clock: src/test_clock.cpp
	hipcc -o bin/test_clock $(CFLAGS) src/test_clock.cpp

clean:
	rm -r obj/
	rm -r bin/

