#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "ANN.h"

#include "Structures.h"
#define INPUT_SIZE 64


int main(char* args) {
	printf("Welcome\n");

	NewAnn(INPUT_SIZE, 300, 2);
	printf("Initialized\n");

	PrintAnn();

	double  data[INPUT_SIZE];
	for (size_t i = 0; i < INPUT_SIZE; i++)
		data[i] = rand() / (double)RAND_MAX;

	Compute(data, INPUT_SIZE);
	printf("Computed\n");
	Ann.LearnRate = 0.001;

	/*double targets[] = { 0.1, 0.2, 0.3 };
	for (size_t i = 0; i < 10; i++)
	{
		Compute(data, 3);
		BackProp(targets, 3);
		PrintOutput();
	}*/

	PrintAnn();

//	FreeANN();
	//free(data);
	int c = _getch();

	return 0;
}