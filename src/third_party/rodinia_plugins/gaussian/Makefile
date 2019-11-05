include ../common.mk
.PHONY: all

all: $(BIN_DIR)/gaussian.so

$(OBJ_DIR)/gaussian.o: gaussian.cpp
	hipcc $(CFLAGS) -c -o $(OBJ_DIR)/gaussian.o gaussian.cpp

$(BIN_DIR)/gaussian.so: $(OBJ_DIR)/gaussian.o
	hipcc --shared $(CFLAGS) -o $(BIN_DIR)/gaussian.so $(OBJ_DIR)/gaussian.o \
		$(PLUGIN_UTILS)

