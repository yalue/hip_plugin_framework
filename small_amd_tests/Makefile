.PHONY: all clean

all: test_timer_granularity test_thread_scheduling

test_timer_granularity: test_timer_granularity.cpp
	hipcc -Wall -Werror -O3 -o test_timer_granularity ./test_timer_granularity.cpp

test_thread_scheduling: test_thread_scheduling.cpp
	hipcc -Wall -Werror -O3 -o test_thread_scheduling ./test_thread_scheduling.cpp

clean:
	rm -f test_timer_granularity
	rm -f test_thread_scheduling

