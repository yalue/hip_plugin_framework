include ../common.mk

HIPFILES := main.cpp dwt.cpp components.cpp dwt_hip/fdwt53.cpp \
	dwt_hip/common.cpp

HEADERS := common.h dwt.h dwt_hip/common.h dwt_hip/dwt.h dwt_hip/io.h \
	dwt_hip/transform_buffer.h dwt_plugin_state.h

$(BIN_DIR)/dwt2d.so: $(HIPFILES) $(HEADERS)
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/dwt2d.so $(HIPFILES) $(PLUGIN_UTILS)

