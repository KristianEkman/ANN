#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <conio.h>
#endif

#include "ANN.h"

static void WaitForExit(void)
{
#ifdef _WIN32
	printf("Done! Press a key to exit");
	(void)_getch();
#else
	printf("Done!\n");
#endif
}

int main(void) {
	const size_t inputCount = 65;
	const size_t hiddenCount = 2000;
	const size_t outputCount = 1;
	ANN* ann;
	double* data;
	double* targets;
	int loops = 1000;
	int exitCode = EXIT_SUCCESS;

	printf("Welcome\n");
	srand((unsigned int)time(NULL));

	ann = NewAnn(inputCount, hiddenCount, outputCount);
	if (ann == NULL)
	{
		fprintf(stderr, "Failed to allocate ANN.\n");
		return EXIT_FAILURE;
	}

	data = malloc(inputCount * sizeof(*data));
	targets = malloc(outputCount * sizeof(*targets));
	if (data == NULL || targets == NULL)
	{
		fprintf(stderr, "Failed to allocate sample buffers.\n");
		free(data);
		free(targets);
		FreeAnn(ann);
		return EXIT_FAILURE;
	}

	printf("Initialized %.2f MB\n", (double)AnnMemoryUsage(ann) / 0x100000);

	for (size_t i = 0; i < inputCount; i++)
		data[i] = (rand() / (double)RAND_MAX) - 1;

	targets[0] = -0.5;
	ann->LearnRate = 0.5;
	clock_t start = clock();
	for (int i = 0; i < loops; i++)
	{
		if (Compute(ann, data, inputCount) != 0 || BackProp(ann, targets, outputCount) != 0)
		{
			fprintf(stderr, "ANN compute step failed.\n");
			exitCode = EXIT_FAILURE;
			goto cleanup;
		}

		PrintOutput(ann);
	}
	clock_t stop = clock();
	double sec = ((double)stop - (double)start) / (double)CLOCKS_PER_SEC;
	printf("Loops per sec: %f\n", loops / sec);
	PrintOutput(ann);

	cleanup:
	free(targets);
	free(data);
	FreeAnn(ann);
	WaitForExit();

	return exitCode;
}
