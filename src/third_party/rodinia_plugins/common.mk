# This contains common paths shared by the Rodinia plugins' independent
# makefiles.

# Jump through hoops to get the top-level directory relative to common.mk,
# rather than whatever includes common.mk.
TOP_DIR := $(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))/../../..)
OBJ_DIR := $(TOP_DIR)/obj
BIN_DIR := $(TOP_DIR)/bin

CFLAGS := -Wall -Werror -O3 -g -fPIC -I$(TOP_DIR)/src
PLUGIN_UTILS := $(OBJ_DIR)/plugin_hip_utilities.o $(OBJ_DIR)/plugin_utilities.o \
	$(OBJ_DIR)/cJSON.o

