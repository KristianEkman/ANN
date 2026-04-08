#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ANN.h"

static void FillInput(double* data)
{
	for (int i = 0; i < INPUT_SIZE - 1; i++)
		data[i] = ((double)(i % 7) / 6.0) - 0.5;
}

static double HalfSquaredError(double target, double value)
{
	double difference = target - value;
	return (difference * difference) / 2.0;
}

static int TestForwardPassIsStable(void)
{
	double data[INPUT_SIZE - 1];
	FillInput(data);

	srand(1);
	NewAnn();
	Compute(data, INPUT_SIZE - 1);
	double firstOutput = Ann.Output[0].Value;

	Compute(data, INPUT_SIZE - 1);
	double secondOutput = Ann.Output[0].Value;

	if (fabs(firstOutput - secondOutput) > 1e-12)
	{
		fprintf(stderr, "Forward pass is not repeatable: %f vs %f\n", firstOutput, secondOutput);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

static int TestTrainingReducesError(void)
{
	double data[INPUT_SIZE - 1];
	double targets[OUTPUT_SIZE] = { 0.25 };
	FillInput(data);

	srand(1);
	NewAnn();
	Ann.LearnRate = 0.05;

	Compute(data, INPUT_SIZE - 1);
	double initialError = HalfSquaredError(targets[0], Ann.Output[0].Value);

	for (int i = 0; i < 250; i++)
	{
		Compute(data, INPUT_SIZE - 1);
		BackProp(targets, OUTPUT_SIZE);
	}

	Compute(data, INPUT_SIZE - 1);
	double finalError = HalfSquaredError(targets[0], Ann.Output[0].Value);

	if (finalError >= initialError)
	{
		fprintf(stderr, "Training did not reduce error: %f -> %f\n", initialError, finalError);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int main(void)
{
	if (TestForwardPassIsStable() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	if (TestTrainingReducesError() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	printf("ANN tests passed\n");
	return EXIT_SUCCESS;
}
