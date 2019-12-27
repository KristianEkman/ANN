#pragma once
#include "Structures.h"

ANN Ann;

void NewAnn(int inputSize, int hiddenSize, int outputSize);
void PrintAnn();
void FreeANN();
void Compute(double* data, int dataLength);
void BackProp(double* targets, int targLength);
void PrintOutput();
