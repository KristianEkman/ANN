#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#include <conio.h>
#endif

#include "ANN.h"

static void WaitForExit(void)
{
#ifdef _WIN32
	printf("Done! Press a key to exit");
	(void)_getch();
#else
	printf("Done!\n");
#endif
}

int main(void) {
	printf("Welcome\n");
	srand((unsigned int)time(NULL));

	NewAnn();
	printf("Initialized %.2f MB\n", (float)sizeof(ANN) / 0x100000 );

	//PrintAnn();
	double  data[INPUT_SIZE - 1];
	for (size_t i = 0; i < INPUT_SIZE - 1; i++)
		data[i] = (rand() / (double)RAND_MAX) - 1;

	Ann.LearnRate = 0.5;
	clock_t start = clock();
	double targets[] = { -0.5 };
	int loops = 1000;
	for (int i = 0; i < loops; i++)
	{
		Compute(data, INPUT_SIZE - 1);
		BackProp(targets, OUTPUT_SIZE);
		PrintOutput();
	}
	clock_t stop = clock();
	double sec = ((double)stop - (double)start) / (double)CLOCKS_PER_SEC;
	//Loops per sec : 580.012760
	printf("Loops per sec: %f\n", loops / sec);
	PrintOutput();
	//PrintAnn();

	WaitForExit();

	return 0;
}
