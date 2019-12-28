#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "ANN.h"
#include <time.h>

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

	Ann.LearnRate = 0.01;
	clock_t start = clock();
	double targets[] = { 0, 1,0,0,0,0,0,0,0,0 };
	int loops = 1000;
	for (size_t i = 0; i < loops; i++)
	{
		Compute(data, INPUT_SIZE);
		BackProp(targets, 10);
		//PrintOutput();
	}
	clock_t stop = clock();
	double sec = (stop - start) / (double)CLOCKS_PER_SEC;
	//Loops per sec: 273.972603
	printf("Loops per sec: %f\n", loops / sec);
	PrintOutput();
	//PrintAnn();

	FreeANN();
	printf("Done! Press a key to exit");
	int c = _getch();

	return 0;
}