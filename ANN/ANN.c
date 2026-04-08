#include "Structures.h"
#include "ANN.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


ANN Ann;

static double RandomWeight(void)
{
	return ((double)rand() / (double)RAND_MAX) - 1.0;
}

static void ResetForwardState(void)
{
	Ann.Inputs[INPUT_SIZE - 1].Value = 1.0;

	for (int h = 0; h < HIDDEN_SIZE - 1; h++)
		Ann.Hidden[h].Value = 0.0;
	Ann.Hidden[HIDDEN_SIZE - 1].Value = 1.0;

	for (int o = 0; o < OUTPUT_SIZE; o++)
		Ann.Output[o].Value = 0.0;
}


void NewAnn(void) {
	memset(&Ann, 0, sizeof Ann);
	Ann.Inputs[INPUT_SIZE - 1].Value = 1.0;
	Ann.Hidden[HIDDEN_SIZE - 1].Value = 1.0;

	for (int i = 0; i < INPUT_SIZE; i++)
	{
		Neuron_I* inputNeuron = &Ann.Inputs[i];
		char inpBias = i == INPUT_SIZE - 1;

		for (int h = 0; h < HIDDEN_SIZE; h++)
		{
			Weight_I_H* pWeightI_H = &inputNeuron->Weights[h];
			pWeightI_H->ConnectedNeuron = &Ann.Hidden[h];
			pWeightI_H->Value = (inpBias && h == HIDDEN_SIZE - 1) ? 0.0 : RandomWeight();
		}
	}

	for (int h = 0; h < HIDDEN_SIZE; h++)
	{
		Neuron_H* hiddenNeuron = &Ann.Hidden[h];
		for (int o = 0; o < OUTPUT_SIZE; o++)
		{
			Weight_H_O* pWeightH_O = &hiddenNeuron->Weights[o];
			pWeightH_O->ConnectedNeuron = &Ann.Output[o];
			pWeightH_O->Value = RandomWeight();
		}
	}

}

void PrintAnn(void) {
	printf("Input\n");
	for (size_t i = 0; i < INPUT_SIZE; i++)
	{
		Neuron_I* neuron = &Ann.Inputs[i];
		printf("%f", neuron->Value);
		for (int n = 0; n < HIDDEN_SIZE; n++)
		{
			Weight_I_H* weight = &neuron->Weights[n];
			printf("\t%f -> %f, ", weight->Value, weight->ConnectedNeuron->Value);
		}
		printf("\n");
	}
	printf("Hidden\n");
	for (size_t i = 0; i < HIDDEN_SIZE; i++)
	{
		Neuron_H* neuron = &Ann.Hidden[i];
		printf("%f", neuron->Value);
		for (int n = 0; n < OUTPUT_SIZE; n++)
		{
			Weight_H_O* weight = &neuron->Weights[n];
			printf("\t%f -> %f, ", weight->Value, weight->ConnectedNeuron->Value);
		}
		printf("\n");
	}
	printf("Otput\n");
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		Neuron_O* neuron = &Ann.Output[i];
		printf("%f\n", neuron->Value);
	}
}

void PrintOutput(void) {
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		Neuron_O* neuron = &Ann.Output[i];
		printf("%f ", neuron->Value);
	}

	printf("   %f", Ann.TotalError);
	printf("\n");
}

double LeakyReLU(double x)
{
	if (x >= 0)
		return x;
	else
		return x / 20;
}

double Sigmoid(double x)
{
	double s = 1 / (1 + exp(-x));
	return s;
}

void Compute(double* data, int dataLength) {

	if (dataLength != INPUT_SIZE - 1)
	{
		fprintf(stderr, "Length of input data is not same as Length of input layer.\n");
		exit(1000);
	}

	ResetForwardState();

	for (int i = 0; i < dataLength; i++)
		Ann.Inputs[i].Value = data[i];

	//Forward propagation
	for (size_t n = 0; n < INPUT_SIZE; n++)
	{
		Neuron_I* neuron = &Ann.Inputs[n];
		for (int i = 0; i < HIDDEN_SIZE - 1; i++)
		{
			Weight_I_H* weight = &neuron->Weights[i];
			weight->ConnectedNeuron->Value += (weight->Value * neuron->Value);
		}
	}

	// Applying aktivation function to all neurons in hidden layer except bias
	for (size_t n = 0; n < HIDDEN_SIZE - 1; n++)
	{
		Neuron_H* neuron = &Ann.Hidden[n];
		neuron->Value = LeakyReLU(neuron->Value / (double)(INPUT_SIZE)); // dividing to prevent over flow.
		//neuron.Value = Sigmoid(neuron.Value / Layers[l].Count);
	}

	//Next layer
	for (size_t n = 0; n < HIDDEN_SIZE; n++)
	{
		Neuron_H* neuron = &Ann.Hidden[n];
		for (size_t i = 0; i < OUTPUT_SIZE; i++)
		{
			Weight_H_O* weight = &neuron->Weights[i];
			weight->ConnectedNeuron->Value += (weight->Value * neuron->Value);
		}
	}

	// Applying aktivation function to all neurons in outputlayer layer
	for (int n = 0; n < OUTPUT_SIZE; n++)
	{
		Neuron_O* neuron = &Ann.Output[n];
		neuron->Value = LeakyReLU(neuron->Value / (double)(HIDDEN_SIZE)); // dividing to prevent over flow.
		//neuron.Value = Sigmoid(neuron.Value / Layers[l].Count);
	}
}

void BackProp(double* targets, int targLength) {

	if (targLength != OUTPUT_SIZE)
	{
		fprintf(stderr, "Length of target data is not same as Length of output layer.\n");
		exit(1000);
	}

	Ann.TotalError = 0;
	//backwards propagation
	for (int n = 0; n < OUTPUT_SIZE; n++)
	{
		Neuron_O* neuron = &Ann.Output[n];
		neuron->Target = targets[n];
		neuron->Error = pow(neuron->Target - neuron->Value, 2) / 2;
		neuron->Delta = (neuron->Value - neuron->Target) * (neuron->Value > 0 ? 1 : 1 / (double)20);
		//neuron.Delta = (neuron->Value - neuron->Target) * (neuron->Value * (1 - neuron->Value));
		Ann.TotalError += neuron->Error;
	}

	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		// This can be done in parallell.
		Neuron_H* neuron = &Ann.Hidden[i];
		for (int w = 0; w < OUTPUT_SIZE; w++)
		{
			Weight_H_O* weight = &neuron->Weights[w];
			weight->Delta = neuron->Value * weight->ConnectedNeuron->Delta;
		}
	}

	//Input Layer
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		//Could be run in parallell
		Neuron_I* neuron = &Ann.Inputs[i];
		for (int w = 0; w < HIDDEN_SIZE - 1; w++)
		{
			Weight_I_H* weight = &neuron->Weights[w];
			weight->Delta = 0.0;
			for (size_t c = 0; c < OUTPUT_SIZE; c++)
			{
				Weight_H_O* connectedWeight = &weight->ConnectedNeuron->Weights[c];
				weight->Delta += connectedWeight->Value * connectedWeight->ConnectedNeuron->Delta;
			}
			double cv = weight->ConnectedNeuron->Value;
			weight->Delta *= cv > 0 ? 1 : 1 / (double)20;
			weight->Delta *= neuron->Value;
		}
	}

	//All deltas are done. Now calculate new weights.
	for (size_t n = 0; n < INPUT_SIZE; n++)
	{
		Neuron_I* neuron = &Ann.Inputs[n];
		for (size_t w = 0; w < HIDDEN_SIZE - 1; w++)
		{
			Weight_I_H* weight = &neuron->Weights[w];
			weight->Value -= (weight->Delta * Ann.LearnRate);
		}
	}

	for (size_t n = 0; n < HIDDEN_SIZE; n++)
	{
		Neuron_H* neuron = &Ann.Hidden[n];
		for (size_t w = 0; w < OUTPUT_SIZE; w++)
		{
			Weight_H_O* weight = &neuron->Weights[w];
			weight->Value -= (weight->Delta * Ann.LearnRate);
		}
	}
}
