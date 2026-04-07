#pragma once
#include "Structures.h"

extern ANN Ann;

void NewAnn(void);
void PrintAnn(void);
void Compute(double* data, int dataLength);
void BackProp(double* targets, int targLength);
void PrintOutput(void);
