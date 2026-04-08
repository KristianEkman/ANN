#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static void PrintUsage(const char* programName)
{
	fprintf(stderr, "Usage: %s [--load FILE] [--save FILE]\n", programName);
}

static int ParseArgs(int argc, char** argv, const char** loadPath, const char** savePath)
{
	*loadPath = NULL;
	*savePath = NULL;

	for (int argumentIndex = 1; argumentIndex < argc; argumentIndex++)
	{
		if (strcmp(argv[argumentIndex], "--help") == 0 || strcmp(argv[argumentIndex], "-h") == 0)
		{
			PrintUsage(argv[0]);
			return 1;
		}

		if (strcmp(argv[argumentIndex], "--load") == 0 || strcmp(argv[argumentIndex], "--save") == 0)
		{
			const int isLoadOption = strcmp(argv[argumentIndex], "--load") == 0;
			const char** targetPath = isLoadOption ? loadPath : savePath;

			if (argumentIndex + 1 >= argc)
			{
				fprintf(stderr, "Missing file path after %s.\n", argv[argumentIndex]);
				PrintUsage(argv[0]);
				return -1;
			}

			if (*targetPath != NULL)
			{
				fprintf(stderr, "%s specified more than once.\n", argv[argumentIndex]);
				PrintUsage(argv[0]);
				return -1;
			}

			*targetPath = argv[++argumentIndex];
			continue;
		}

		fprintf(stderr, "Unknown argument: %s\n", argv[argumentIndex]);
		PrintUsage(argv[0]);
		return -1;
	}

	return 0;
}

int main(int argc, char** argv) {
	const size_t inputCount = 65;
	const size_t hiddenCount = 2000;
	const size_t outputCount = 1;
	const char* loadPath;
	const char* savePath;
	ANN* ann;
	double* data;
	double* targets;
	int loops = 1000;
	int exitCode = EXIT_SUCCESS;
	int parseStatus;

	parseStatus = ParseArgs(argc, argv, &loadPath, &savePath);
	if (parseStatus != 0)
		return parseStatus > 0 ? EXIT_SUCCESS : EXIT_FAILURE;

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
	if (loadPath != NULL)
	{
		if (LoadAnnWeights(ann, loadPath) != 0)
		{
			fprintf(stderr, "Failed to load weights from %s.\n", loadPath);
			exitCode = EXIT_FAILURE;
			goto cleanup;
		}

		printf("Loaded weights from %s\n", loadPath);
	}

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

	if (savePath != NULL)
	{
		if (SaveAnnWeights(ann, savePath) != 0)
		{
			fprintf(stderr, "Failed to save weights to %s.\n", savePath);
			exitCode = EXIT_FAILURE;
			goto cleanup;
		}

		printf("Saved weights to %s\n", savePath);
	}

	cleanup:
	free(targets);
	free(data);
	FreeAnn(ann);
	WaitForExit();

	return exitCode;
}
