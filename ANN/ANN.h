#pragma once
#include "Structures.h"

ANN* NewAnn(size_t inputCount, size_t hiddenCount, size_t outputCount);
void FreeAnn(ANN* ann);
size_t AnnMemoryUsage(const ANN* ann);
void PrintAnn(const ANN* ann);
int Compute(ANN* ann, const double* data, size_t dataLength);
int BackProp(ANN* ann, const double* targets, size_t targLength);
void PrintOutput(const ANN* ann);
