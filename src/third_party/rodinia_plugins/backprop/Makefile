include ../common.mk
.PHONY: all

HIPFILES := backprop.cpp backprop_plugin.cpp backprop_hip_kernel.cpp

# Create one .o file in $(OBJ_DIR) for each of the files in HIPFILES.
HIPOBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(HIPFILES))

all: $(BIN_DIR)/backprop.so

$(OBJ_DIR)/%.o: %.cpp
	hipcc $(CFLAGS) -c -o $@ $<

$(BIN_DIR)/backprop.so: $(HIPOBJ)
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/backprop.so $(HIPOBJ) \
		$(PLUGIN_UTILS)

