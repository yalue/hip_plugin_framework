.PHONY: all clean plugins directories

CFLAGS := -Wall -Werror -O3 -g -fPIC

all: directories bin/runner

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

bin/runner: obj/runner.o obj/cJSON.o obj/parse_config.o obj/barrier_wait.o
	gcc $(CFLAGS) -o bin/runner obj/runner.o obj/cJSON.o obj/parse_config.o \
		obj/barrier_wait.o -ldl -lpthread

clean:
	rm -r obj/
	rm -r bin/
