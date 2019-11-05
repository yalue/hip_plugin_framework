include ../common.mk
.PHONY: all

HIPFILES := dwt_main.cpp dwt.cpp dwt_components.cpp fdwt53.cpp

HEADERS := common.h dwt.h hip_common.h hip_dwt.h hip_io.h \
	hip_transform_buffer.h dwt_plugin_state.h

# I'm not very good at make, but this *should* just be telling it to create one
# .o file for each .cpp file in $(OBJ_DIR).
HIPOBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(HIPFILES))

all: $(BIN_DIR)/dwt2d.so

$(OBJ_DIR)/%.o: %.cpp
	hipcc -c $(CFLAGS) -o $@ $<

$(BIN_DIR)/dwt2d.so: $(HIPOBJ)
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/dwt2d.so $(HIPOBJ) $(PLUGIN_UTILS)

