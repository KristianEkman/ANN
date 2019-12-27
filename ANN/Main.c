#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "ANN.h"

#include "Structures.h"

int main(char* args) {
	printf("Welcome to ANN\n");

	NewAnn(3, 30, 2);
	printf("Initialized\n");

#ifdef _DEBUG
	PrintAnn();
#endif

	float data[] = { 4, 5, 6 };
	Compute(data, 3);
	printf("Computed\n");

#ifdef _DEBUG
	PrintAnn();
#endif

	FreeANN();
	int c = _getch();

	return 0;
}