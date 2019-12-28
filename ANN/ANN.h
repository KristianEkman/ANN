#pragma once
#include "Structures.h"

ANN Ann;

void NewAnn();
void PrintAnn();
void Compute(double* data, int dataLength);
void BackProp(double* targets, int targLength);
void PrintOutput();
