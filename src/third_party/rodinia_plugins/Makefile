include common.mk

.PHONY: all gaussian backprop particlefilter

# This makefile assumes that plugin_utilities.o and plugin_hip_utilities.o
# already exist and are already at the correct location. This will be the case
# if you run "make rodinia_plugins" from the top-level directory. See
# ./common.mk for paths and common dependency definitions.

# These prerequisites intentionally don't name "real" files so that we'll
# always descend into these directories and run make--this allows them to be
# built in parallel, and the makefiles in each individual benchmark's directory
# are responsible for "checking" whether they need to be rebuilt.
all: gaussian backprop particlefilter

gaussian:
	cd gaussian && make

backprop:
	cd backprop && make

particlefilter:
	cd particlefilter && make

