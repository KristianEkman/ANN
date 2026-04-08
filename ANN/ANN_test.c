#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ANN.h"

enum {
	SMALL_INPUT_COUNT = 5,
	SMALL_HIDDEN_COUNT = 7,
	SMALL_OUTPUT_COUNT = 2,
	TRAIN_INPUT_COUNT = 6,
	TRAIN_HIDDEN_COUNT = 9,
	TRAIN_OUTPUT_COUNT = 2,
};

static void FillInput(double* data, size_t inputCount)
{
	for (size_t i = 0; i < inputCount; i++)
		data[i] = ((double)(i % 7) / 6.0) - 0.5;
}

static double TotalHalfSquaredError(const double* targets, const double* values, size_t outputCount)
{
	double total = 0.0;

	for (size_t i = 0; i < outputCount; i++)
	{
		double difference = targets[i] - values[i];
		total += (difference * difference) / 2.0;
	}

	return total;
}

static void FillWeights(ANN* ann)
{
	for (size_t i = 0; i < (ann->InputCount + 1) * ann->HiddenCount; i++)
		ann->WeightsInputHidden[i] = ((double)(i + 1) / 17.0) - 0.75;

	for (size_t i = 0; i < (ann->HiddenCount + 1) * ann->OutputCount; i++)
		ann->WeightsHiddenOutput[i] = ((double)(i + 3) / 19.0) - 0.5;
}

static int DoublesMatch(const double* left, const double* right, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		if (fabs(left[i] - right[i]) > 1e-12)
			return 0;
	}

	return 1;
}

static int TestRejectsZeroSizedLayer(void)
{
	ANN* ann = NewAnn(0, 4, 1);

	if (ann != NULL)
	{
		fprintf(stderr, "ANN should reject zero-sized layers.\n");
		FreeAnn(ann);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

static int TestForwardPassIsStable(void)
{
	ANN* ann;
	double data[SMALL_INPUT_COUNT];
	double firstOutput[SMALL_OUTPUT_COUNT];

	FillInput(data, SMALL_INPUT_COUNT);

	srand(1);
	ann = NewAnn(SMALL_INPUT_COUNT, SMALL_HIDDEN_COUNT, SMALL_OUTPUT_COUNT);
	if (ann == NULL)
	{
		fprintf(stderr, "Failed to allocate ANN for stability test.\n");
		return EXIT_FAILURE;
	}

	if (Compute(ann, data, SMALL_INPUT_COUNT) != 0)
	{
		fprintf(stderr, "Initial compute failed in stability test.\n");
		FreeAnn(ann);
		return EXIT_FAILURE;
	}

	memcpy(firstOutput, ann->Output, sizeof(firstOutput));

	if (Compute(ann, data, SMALL_INPUT_COUNT) != 0)
	{
		fprintf(stderr, "Repeated compute failed in stability test.\n");
		FreeAnn(ann);
		return EXIT_FAILURE;
	}

	for (size_t outputIndex = 0; outputIndex < SMALL_OUTPUT_COUNT; outputIndex++)
	{
		if (fabs(firstOutput[outputIndex] - ann->Output[outputIndex]) > 1e-12)
		{
			fprintf(stderr, "Forward pass is not repeatable at output %zu: %f vs %f\n",
				outputIndex,
				firstOutput[outputIndex],
				ann->Output[outputIndex]);
			FreeAnn(ann);
			return EXIT_FAILURE;
		}
	}

	FreeAnn(ann);

	return EXIT_SUCCESS;
}

static int TestTrainingReducesError(void)
{
	ANN* ann;
	double data[TRAIN_INPUT_COUNT];
	double targets[TRAIN_OUTPUT_COUNT] = { 0.25, -0.15 };
	double initialError;
	double finalError;

	FillInput(data, TRAIN_INPUT_COUNT);

	srand(1);
	ann = NewAnn(TRAIN_INPUT_COUNT, TRAIN_HIDDEN_COUNT, TRAIN_OUTPUT_COUNT);
	if (ann == NULL)
	{
		fprintf(stderr, "Failed to allocate ANN for training test.\n");
		return EXIT_FAILURE;
	}
	ann->LearnRate = 0.05;

	if (Compute(ann, data, TRAIN_INPUT_COUNT) != 0)
	{
		fprintf(stderr, "Initial compute failed in training test.\n");
		FreeAnn(ann);
		return EXIT_FAILURE;
	}
	initialError = TotalHalfSquaredError(targets, ann->Output, TRAIN_OUTPUT_COUNT);

	for (int i = 0; i < 400; i++)
	{
		if (Compute(ann, data, TRAIN_INPUT_COUNT) != 0 || BackProp(ann, targets, TRAIN_OUTPUT_COUNT) != 0)
		{
			fprintf(stderr, "Training step failed.\n");
			FreeAnn(ann);
			return EXIT_FAILURE;
		}
	}

	if (Compute(ann, data, TRAIN_INPUT_COUNT) != 0)
	{
		fprintf(stderr, "Final compute failed in training test.\n");
		FreeAnn(ann);
		return EXIT_FAILURE;
	}
	finalError = TotalHalfSquaredError(targets, ann->Output, TRAIN_OUTPUT_COUNT);

	if (finalError >= initialError)
	{
		fprintf(stderr, "Training did not reduce error: %f -> %f\n", initialError, finalError);
		FreeAnn(ann);
		return EXIT_FAILURE;
	}

	FreeAnn(ann);

	return EXIT_SUCCESS;
}

static int TestSaveAndLoadWeightsRoundTrip(void)
{
	const char* weightsPath = "ann_weights_round_trip.txt";
	ANN* source;
	ANN* loaded;
	double data[SMALL_INPUT_COUNT];

	FillInput(data, SMALL_INPUT_COUNT);
	remove(weightsPath);

	source = NewAnn(SMALL_INPUT_COUNT, SMALL_HIDDEN_COUNT, SMALL_OUTPUT_COUNT);
	loaded = NewAnn(SMALL_INPUT_COUNT, SMALL_HIDDEN_COUNT, SMALL_OUTPUT_COUNT);
	if (source == NULL || loaded == NULL)
	{
		fprintf(stderr, "Failed to allocate ANN for weight persistence test.\n");
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	FillWeights(source);
	loaded->LearnRate = 0.25;

	if (SaveAnnWeights(source, weightsPath) != 0)
	{
		fprintf(stderr, "Failed to save weights.\n");
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	loaded->Inputs[0] = 42.0;
	loaded->Hidden[0] = -13.0;
	loaded->Output[0] = 9.0;
	loaded->TotalError = 7.0;

	if (LoadAnnWeights(loaded, weightsPath) != 0)
	{
		fprintf(stderr, "Failed to load saved weights.\n");
		remove(weightsPath);
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	if (!DoublesMatch(source->WeightsInputHidden,
		loaded->WeightsInputHidden,
		(SMALL_INPUT_COUNT + 1) * SMALL_HIDDEN_COUNT)
		|| !DoublesMatch(source->WeightsHiddenOutput,
			loaded->WeightsHiddenOutput,
			(SMALL_HIDDEN_COUNT + 1) * SMALL_OUTPUT_COUNT))
	{
		fprintf(stderr, "Loaded weights do not match saved weights.\n");
		remove(weightsPath);
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	if (loaded->Inputs[0] != 0.0 || loaded->Hidden[0] != 0.0 || loaded->Output[0] != 0.0 || loaded->TotalError != 0.0)
	{
		fprintf(stderr, "Loading weights should reset transient ANN state.\n");
		remove(weightsPath);
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	if (Compute(source, data, SMALL_INPUT_COUNT) != 0 || Compute(loaded, data, SMALL_INPUT_COUNT) != 0)
	{
		fprintf(stderr, "Compute failed after loading weights.\n");
		remove(weightsPath);
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	if (!DoublesMatch(source->Output, loaded->Output, SMALL_OUTPUT_COUNT))
	{
		fprintf(stderr, "Loaded ANN output does not match saved ANN output.\n");
		remove(weightsPath);
		FreeAnn(loaded);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	remove(weightsPath);
	FreeAnn(loaded);
	FreeAnn(source);
	return EXIT_SUCCESS;
}

static int TestRejectsMismatchedWeightFile(void)
{
	const char* weightsPath = "ann_weights_mismatch.txt";
	ANN* source;
	ANN* mismatched;
	double originalWeight;

	remove(weightsPath);

	source = NewAnn(SMALL_INPUT_COUNT, SMALL_HIDDEN_COUNT, SMALL_OUTPUT_COUNT);
	mismatched = NewAnn(SMALL_INPUT_COUNT + 1, SMALL_HIDDEN_COUNT, SMALL_OUTPUT_COUNT);
	if (source == NULL || mismatched == NULL)
	{
		fprintf(stderr, "Failed to allocate ANN for mismatched weight test.\n");
		FreeAnn(mismatched);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	FillWeights(source);
	originalWeight = mismatched->WeightsInputHidden[0];

	if (SaveAnnWeights(source, weightsPath) != 0)
	{
		fprintf(stderr, "Failed to save weights for mismatch test.\n");
		FreeAnn(mismatched);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	if (LoadAnnWeights(mismatched, weightsPath) == 0)
	{
		fprintf(stderr, "Loading should fail when ANN dimensions do not match.\n");
		remove(weightsPath);
		FreeAnn(mismatched);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	if (mismatched->WeightsInputHidden[0] != originalWeight)
	{
		fprintf(stderr, "Failed load should not modify existing weights.\n");
		remove(weightsPath);
		FreeAnn(mismatched);
		FreeAnn(source);
		return EXIT_FAILURE;
	}

	remove(weightsPath);
	FreeAnn(mismatched);
	FreeAnn(source);
	return EXIT_SUCCESS;
}

int main(void)
{
	if (TestRejectsZeroSizedLayer() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	if (TestForwardPassIsStable() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	if (TestTrainingReducesError() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	if (TestSaveAndLoadWeightsRoundTrip() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	if (TestRejectsMismatchedWeightFile() != EXIT_SUCCESS)
		return EXIT_FAILURE;

	printf("ANN tests passed\n");
	return EXIT_SUCCESS;
}
