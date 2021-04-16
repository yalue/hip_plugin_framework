include ../common.mk
.PHONY: all

all: $(BIN_DIR)/particlefilter_naive.so

$(OBJ_DIR)/particlefilter_naive.o: particlefilter_naive.cpp
	hipcc $(CFLAGS) -c -o $(OBJ_DIR)/particlefilter_naive.o \
		particlefilter_naive.cpp

$(BIN_DIR)/particlefilter_naive.so: $(OBJ_DIR)/particlefilter_naive.o
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/particlefilter_naive.so \
		$(OBJ_DIR)/particlefilter_naive.o $(PLUGIN_UTILS)

