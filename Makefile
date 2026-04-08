BUILD_DIR := build
CMAKE := cmake
CTEST := ctest
RUN_TARGET := $(BUILD_DIR)/ANN

.PHONY: all rebuild configure build test run run-test clean

all: rebuild

rebuild: clean configure build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

build:
	$(CMAKE) --build $(BUILD_DIR)

test: rebuild run-test

run: rebuild
	./$(RUN_TARGET)

run-test:
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)