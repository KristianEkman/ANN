#pragma once
#include "Structures.h"

ANN Ann;

void NewAnn(int inputSize, int hiddenSize, int outputSize);
void PrintAnn();
void FreeANN();
void Compute(float* data, int dataLength);
