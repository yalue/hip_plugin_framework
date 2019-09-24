include ../common.mk

$(BIN_DIR)/backprop.so: backprop.cpp backprop.h backprop_plugin.cpp \
	backprop_hip_kernel.cpp backprop_state.h
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/backprop.so backprop.cpp \
		backprop_plugin.cpp $(PLUGIN_UTILS)

