include ../common.mk

$(BIN_DIR)/gaussian.so: gaussian.cpp
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/gaussian.so gaussian.cpp \
		$(PLUGIN_UTILS)

