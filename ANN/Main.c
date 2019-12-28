#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>

#include "ANN.h"
#include "Structures.h"

int main(char* args) {
	printf("Welcome\n");

	NewAnn();
	printf("Initialized %.2f MB\n", (float)sizeof(ANN) / 0x100000 );

	//PrintAnn();
	double  data[INPUT_SIZE - 1];
	for (size_t i = 0; i < INPUT_SIZE - 1; i++)
		data[i] = rand() / (double)RAND_MAX;

	Ann.LearnRate = 0.5;
	clock_t start = clock();
	double targets[] = { 0, 1,0,0,0,0,0,0,0,0 };
	int loops = 100;
	for (size_t i = 0; i < loops; i++)
	{
		Compute(data, INPUT_SIZE - 1);
		BackProp(targets, 10);
		PrintOutput();
	}
	clock_t stop = clock();
	double sec = ((double)stop - (double)start) / (double)CLOCKS_PER_SEC;
	//Loops per sec : 580.012760
	printf("Loops per sec: %f\n", loops / sec);
	PrintOutput();
	//PrintAnn();

	printf("Done! Press a key to exit");
	int c = _getch();

	return 0;
}