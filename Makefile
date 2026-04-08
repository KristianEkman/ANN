BUILD_DIR := build
CMAKE := cmake
CTEST := ctest

.PHONY: all rebuild configure build test run-test clean

all: rebuild

rebuild: clean configure build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

build:
	$(CMAKE) --build $(BUILD_DIR)

test: rebuild run-test

run-test:
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)