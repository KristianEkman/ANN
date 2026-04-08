#include "ANN.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t InputNodeCount(const ANN* ann)
{
	return ann->InputCount + 1;
}

static size_t HiddenNodeCount(const ANN* ann)
{
	return ann->HiddenCount + 1;
}

static size_t InputHiddenWeightIndex(const ANN* ann, size_t inputIndex, size_t hiddenIndex)
{
	return (inputIndex * ann->HiddenCount) + hiddenIndex;
}

static size_t HiddenOutputWeightIndex(const ANN* ann, size_t hiddenIndex, size_t outputIndex)
{
	return (hiddenIndex * ann->OutputCount) + outputIndex;
}

static size_t InputHiddenWeightCount(const ANN* ann)
{
	return InputNodeCount(ann) * ann->HiddenCount;
}

static size_t HiddenOutputWeightCount(const ANN* ann)
{
	return HiddenNodeCount(ann) * ann->OutputCount;
}

static int AllocateDoubles(double** buffer, size_t count)
{
	*buffer = calloc(count, sizeof(**buffer));
	return *buffer != NULL;
}

static double RandomWeight(void)
{
	return ((double)rand() / (double)RAND_MAX) - 1.0;
}

static double LeakyReLU(double x)
{
	if (x >= 0.0)
		return x;

	return x / 20.0;
}

static double LeakyReLUDerivative(double x)
{
	if (x >= 0.0)
		return 1.0;

	return 1.0 / 20.0;
}

static void RandomizeWeights(ANN* ann)
{
	for (size_t i = 0; i < InputHiddenWeightCount(ann); i++)
		ann->WeightsInputHidden[i] = RandomWeight();

	for (size_t i = 0; i < HiddenOutputWeightCount(ann); i++)
		ann->WeightsHiddenOutput[i] = RandomWeight();
}

ANN* NewAnn(size_t inputCount, size_t hiddenCount, size_t outputCount)
{
	ANN* ann;

	if (inputCount == 0 || hiddenCount == 0 || outputCount == 0)
		return NULL;

	ann = calloc(1, sizeof(*ann));
	if (ann == NULL)
		return NULL;

	ann->InputCount = inputCount;
	ann->HiddenCount = hiddenCount;
	ann->OutputCount = outputCount;

	if (!AllocateDoubles(&ann->Inputs, InputNodeCount(ann))
		|| !AllocateDoubles(&ann->Hidden, HiddenNodeCount(ann))
		|| !AllocateDoubles(&ann->Output, ann->OutputCount)
		|| !AllocateDoubles(&ann->OutputTargets, ann->OutputCount)
		|| !AllocateDoubles(&ann->OutputErrors, ann->OutputCount)
		|| !AllocateDoubles(&ann->OutputDelta, ann->OutputCount)
		|| !AllocateDoubles(&ann->HiddenDelta, ann->HiddenCount)
		|| !AllocateDoubles(&ann->WeightsInputHidden, InputHiddenWeightCount(ann))
		|| !AllocateDoubles(&ann->DeltaInputHidden, InputHiddenWeightCount(ann))
		|| !AllocateDoubles(&ann->WeightsHiddenOutput, HiddenOutputWeightCount(ann))
		|| !AllocateDoubles(&ann->DeltaHiddenOutput, HiddenOutputWeightCount(ann)))
	{
		FreeAnn(ann);
		return NULL;
	}

	ann->Inputs[ann->InputCount] = 1.0;
	ann->Hidden[ann->HiddenCount] = 1.0;
	RandomizeWeights(ann);

	return ann;
}

void FreeAnn(ANN* ann)
{
	if (ann == NULL)
		return;

	free(ann->Inputs);
	free(ann->Hidden);
	free(ann->Output);
	free(ann->OutputTargets);
	free(ann->OutputErrors);
	free(ann->OutputDelta);
	free(ann->HiddenDelta);
	free(ann->WeightsInputHidden);
	free(ann->DeltaInputHidden);
	free(ann->WeightsHiddenOutput);
	free(ann->DeltaHiddenOutput);
	free(ann);
}

size_t AnnMemoryUsage(const ANN* ann)
{
	size_t totalBytes;

	if (ann == NULL)
		return 0;

	totalBytes = sizeof(*ann);
	totalBytes += InputNodeCount(ann) * sizeof(*ann->Inputs);
	totalBytes += HiddenNodeCount(ann) * sizeof(*ann->Hidden);
	totalBytes += ann->OutputCount * sizeof(*ann->Output);
	totalBytes += ann->OutputCount * sizeof(*ann->OutputTargets);
	totalBytes += ann->OutputCount * sizeof(*ann->OutputErrors);
	totalBytes += ann->OutputCount * sizeof(*ann->OutputDelta);
	totalBytes += ann->HiddenCount * sizeof(*ann->HiddenDelta);
	totalBytes += InputHiddenWeightCount(ann) * sizeof(*ann->WeightsInputHidden);
	totalBytes += InputHiddenWeightCount(ann) * sizeof(*ann->DeltaInputHidden);
	totalBytes += HiddenOutputWeightCount(ann) * sizeof(*ann->WeightsHiddenOutput);
	totalBytes += HiddenOutputWeightCount(ann) * sizeof(*ann->DeltaHiddenOutput);

	return totalBytes;
}

void PrintAnn(const ANN* ann)
{
	if (ann == NULL)
		return;

	printf("Input\n");
	for (size_t i = 0; i < InputNodeCount(ann); i++)
	{
		printf("%f", ann->Inputs[i]);
		for (size_t h = 0; h < ann->HiddenCount; h++)
		{
			size_t index = InputHiddenWeightIndex(ann, i, h);
			printf("\t%f -> %f, ", ann->WeightsInputHidden[index], ann->Hidden[h]);
		}
		printf("\n");
	}

	printf("Hidden\n");
	for (size_t h = 0; h < HiddenNodeCount(ann); h++)
	{
		printf("%f", ann->Hidden[h]);
		for (size_t o = 0; o < ann->OutputCount; o++)
		{
			size_t index = HiddenOutputWeightIndex(ann, h, o);
			printf("\t%f -> %f, ", ann->WeightsHiddenOutput[index], ann->Output[o]);
		}
		printf("\n");
	}

	printf("Output\n");
	for (size_t o = 0; o < ann->OutputCount; o++)
		printf("%f\n", ann->Output[o]);
}

void PrintOutput(const ANN* ann)
{
	if (ann == NULL)
		return;

	for (size_t i = 0; i < ann->OutputCount; i++)
		printf("%f ", ann->Output[i]);

	printf("   %f\n", ann->TotalError);
}

int Compute(ANN* ann, const double* data, size_t dataLength)
{
	if (ann == NULL || data == NULL || dataLength != ann->InputCount)
		return -1;

	memcpy(ann->Inputs, data, ann->InputCount * sizeof(*ann->Inputs));
	ann->Inputs[ann->InputCount] = 1.0;

	memset(ann->Hidden, 0, HiddenNodeCount(ann) * sizeof(*ann->Hidden));
	for (size_t i = 0; i < InputNodeCount(ann); i++)
	{
		double inputValue = ann->Inputs[i];
		double* weightsRow = &ann->WeightsInputHidden[i * ann->HiddenCount];
		for (size_t h = 0; h < ann->HiddenCount; h++)
			ann->Hidden[h] += weightsRow[h] * inputValue;
	}

	for (size_t h = 0; h < ann->HiddenCount; h++)
		ann->Hidden[h] = LeakyReLU(ann->Hidden[h] / (double)InputNodeCount(ann));
	ann->Hidden[ann->HiddenCount] = 1.0;

	memset(ann->Output, 0, ann->OutputCount * sizeof(*ann->Output));
	for (size_t h = 0; h < HiddenNodeCount(ann); h++)
	{
		double hiddenValue = ann->Hidden[h];
		double* weightsRow = &ann->WeightsHiddenOutput[h * ann->OutputCount];
		for (size_t o = 0; o < ann->OutputCount; o++)
			ann->Output[o] += weightsRow[o] * hiddenValue;
	}

	for (size_t o = 0; o < ann->OutputCount; o++)
		ann->Output[o] = LeakyReLU(ann->Output[o] / (double)HiddenNodeCount(ann));

	return 0;
}

int BackProp(ANN* ann, const double* targets, size_t targLength)
{
	if (ann == NULL || targets == NULL || targLength != ann->OutputCount)
		return -1;

	ann->TotalError = 0.0;
	for (size_t o = 0; o < ann->OutputCount; o++)
	{
		double difference = targets[o] - ann->Output[o];

		ann->OutputTargets[o] = targets[o];
		ann->OutputErrors[o] = (difference * difference) / 2.0;
		ann->OutputDelta[o] = (ann->Output[o] - targets[o]) * LeakyReLUDerivative(ann->Output[o]);
		ann->TotalError += ann->OutputErrors[o];
	}

	for (size_t h = 0; h < ann->HiddenCount; h++)
	{
		double downstream = 0.0;
		for (size_t o = 0; o < ann->OutputCount; o++)
		{
			size_t index = HiddenOutputWeightIndex(ann, h, o);
			downstream += ann->WeightsHiddenOutput[index] * ann->OutputDelta[o];
			ann->DeltaHiddenOutput[index] = ann->Hidden[h] * ann->OutputDelta[o];
		}

		ann->HiddenDelta[h] = downstream * LeakyReLUDerivative(ann->Hidden[h]);
	}

	for (size_t o = 0; o < ann->OutputCount; o++)
	{
		size_t biasIndex = HiddenOutputWeightIndex(ann, ann->HiddenCount, o);
		ann->DeltaHiddenOutput[biasIndex] = ann->Hidden[ann->HiddenCount] * ann->OutputDelta[o];
	}

	for (size_t i = 0; i < InputNodeCount(ann); i++)
	{
		double inputValue = ann->Inputs[i];
		double* deltaRow = &ann->DeltaInputHidden[i * ann->HiddenCount];
		for (size_t h = 0; h < ann->HiddenCount; h++)
			deltaRow[h] = inputValue * ann->HiddenDelta[h];
	}

	for (size_t i = 0; i < InputHiddenWeightCount(ann); i++)
		ann->WeightsInputHidden[i] -= ann->DeltaInputHidden[i] * ann->LearnRate;

	for (size_t i = 0; i < HiddenOutputWeightCount(ann); i++)
		ann->WeightsHiddenOutput[i] -= ann->DeltaHiddenOutput[i] * ann->LearnRate;

	return 0;
}
