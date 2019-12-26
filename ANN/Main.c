#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "ANN.h"

#include "Structures.h"

int main(char* args) {
	

	NewAnn(64, 1000, 1);

	PrintAnn();

	FreeANN();
	int c = _getch();

	return 0;
}