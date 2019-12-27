#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "ANN.h"

#include "Structures.h"
#define INPUT_SIZE 728

int main(char* args) {
	printf("Welcome\n");

	NewAnn(INPUT_SIZE, 200, 10);
	printf("Initialized\n");

	//PrintAnn();

	double  data[INPUT_SIZE];
	for (size_t i = 0; i < INPUT_SIZE; i++)
		data[i] = rand() / (double)RAND_MAX;

	Compute(data, INPUT_SIZE);
	printf("Computed\n");
	Ann.LearnRate = 0.01;

	double targets[] = { 0, 1,0,0,0,0,0,0,0,0 };
	for (size_t i = 0; i < 500; i++)
	{
		Compute(data, INPUT_SIZE);
		BackProp(targets, 10);
		//PrintOutput();
	}

	//PrintAnn();

//	FreeANN();
	//free(data);
	printf("Done! Press a key to exit");
	int c = _getch();

	return 0;
}