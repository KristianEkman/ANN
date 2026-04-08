# ANN

Small feed-forward neural network library written in C.

The repository builds three outputs:

- `ANN_lib`: reusable static library target
- `ANN`: sample executable in `ANN/Main.c`
- `ANN_test`: test executable in `ANN/ANN_test.c`

The canonical build is defined in `CMakeLists.txt`. The root `Makefile` is a convenience wrapper around the same CMake build.

## Requirements

- CMake 3.16 or newer
- A C11 compiler
- `make` if you want to use the wrapper commands on macOS or Linux

## Build

### Quick build with Make

From the repository root:

```sh
make
```

This performs a clean rebuild by running:

1. `cmake -S . -B build`
2. `cmake --build build`

Useful Make targets:

- `make`: clean, configure, and build
- `make build`: build the existing `build` directory
- `make run`: rebuild and run the sample program
- `make test`: rebuild and run the test suite
- `make run-test`: run tests in the existing `build` directory
- `make clean`: remove the `build` directory

### Direct CMake build

From the repository root:

```sh
cmake -S . -B build
cmake --build build
```

On macOS and Linux, the static library is produced as:

```sh
build/libANN_lib.a
```

The sample executable is produced as:

```sh
build/ANN
```

The test executable is produced as:

```sh
build/ANN_test
```

## Run

Run the sample application:

```sh
make run
```

Or run it directly after building:

```sh
./build/ANN
```

To load or save a weights file with the sample executable:

```sh
./build/ANN --load weights.txt --save weights.txt
```

The Make wrapper also passes optional arguments through:

```sh
make run ARGS="--load weights.txt --save weights.txt"
```

## Test

Run the full test suite:

```sh
make test
```

Or with CTest directly:

```sh
ctest --test-dir build --output-on-failure
```

## Use As A Library

Yes, this project can be used from another C program.

The public header is:

```c
#include "ANN.h"
```

The library API currently exposes:

- `ANN* NewAnn(size_t inputCount, size_t hiddenCount, size_t outputCount)`
- `void FreeAnn(ANN* ann)`
- `size_t AnnMemoryUsage(const ANN* ann)`
- `int Compute(ANN* ann, const double* data, size_t dataLength)`
- `int BackProp(ANN* ann, const double* targets, size_t targLength)`
- `void PrintAnn(const ANN* ann)`
- `void PrintOutput(const ANN* ann)`
- `int SaveAnnWeights(const ANN* ann, const char* filePath)`
- `int LoadAnnWeights(ANN* ann, const char* filePath)`

The `ANN` struct is public, so runtime parameters such as `LearnRate` are configured directly on the instance.

### Minimal example

```c
#include <stdlib.h>
#include <time.h>

#include "ANN.h"

int main(void)
{
	ANN* ann;
	double input[3] = { 0.1, -0.2, 0.3 };
	double target[1] = { 0.4 };

	srand((unsigned)time(NULL));

	ann = NewAnn(3, 8, 1);
	if (ann == NULL)
		return 1;

	ann->LearnRate = 0.05;

	if (Compute(ann, input, 3) != 0)
	{
		FreeAnn(ann);
		return 1;
	}

	if (BackProp(ann, target, 1) != 0)
	{
		FreeAnn(ann);
		return 1;
	}

	if (SaveAnnWeights(ann, "weights.txt") != 0)
	{
		FreeAnn(ann);
		return 1;
	}

	if (LoadAnnWeights(ann, "weights.txt") != 0)
	{
		FreeAnn(ann);
		return 1;
	}

	FreeAnn(ann);
	return 0;
}
```

### Link from another CMake project

If another CMake project wants to use this library directly from source:

```cmake
add_subdirectory(/path/to/ANN ann_build)
target_link_libraries(your_app PRIVATE ANN_lib)
```

Because `ANN_lib` publishes the `ANN` include directory, `#include "ANN.h"` works in the consuming target.

### Link against the built static library

If you already built this repository, you can link your own program against the generated archive:

```sh
cc your_program.c build/libANN_lib.a -IANN -lm
```

On Unix-like systems the math library flag `-lm` is required.

## Notes

- `NewAnn` initializes weights with `rand()`, so call `srand(...)` before creating a network if you want non-deterministic initialization.
- Weight files are stored as plain text with a version header and the expected layer sizes. `LoadAnnWeights(...)` rejects files that do not match the ANN dimensions.
- There is currently no `install()` step or `find_package()` support. Reuse is done either by `add_subdirectory(...)` or by linking the generated static library directly.
