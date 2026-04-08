#pragma once

#include <stddef.h>

typedef struct {
	size_t InputCount;
	size_t HiddenCount;
	size_t OutputCount;

	double* Inputs;
	double* Hidden;
	double* Output;

	double* OutputTargets;
	double* OutputErrors;
	double* OutputDelta;
	double* HiddenDelta;

	double* WeightsInputHidden;
	double* DeltaInputHidden;
	double* WeightsHiddenOutput;
	double* DeltaHiddenOutput;

	double TotalError;
	double LearnRate;
} ANN;

