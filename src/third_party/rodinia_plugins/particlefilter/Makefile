include ../common.mk
.PHONY: all

# TODO: Add particlefilter_float, maybe?
all: $(BIN_DIR)/particlefilter_naive.so

$(BIN_DIR)/particlefilter_naive.so: particlefilter_naive.cpp
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/particlefilter_naive.so \
		particlefilter_naive.cpp $(PLUGIN_UTILS)

