#include "Structures.h"
#include "ANN.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Windows.h>
#include <math.h>


void NewAnn(int inputSize, int hiddenSize, int outputSize) {
	NeuronsList* input = &Ann.Layers[0];
	input->Items = GlobalAlloc(0, (inputSize + 1) * sizeof(Neuron));//allocating for bias
	input->Length = inputSize + 1;

	NeuronsList* hidden = &Ann.Layers[1];
	hidden->Items = GlobalAlloc(0, (hiddenSize + 1) * sizeof(Neuron));//allocating for bias
	hidden->Length = hiddenSize + 1;

	NeuronsList* output = &Ann.Layers[2];
	output->Items = GlobalAlloc(0, (outputSize) * sizeof(Neuron));
	output->Length = outputSize;

	for (int i = 0; i < input->Length; i++)
	{
		Neuron* inputNeuron = &input->Items[i];
		char inpBias = i == input->Length - 1;
		inputNeuron->Value = inpBias ? 1 : 0;

		inputNeuron->Weights = GlobalAlloc(0, sizeof(WeightsList));
		inputNeuron->Weights->Items = GlobalAlloc(0, (hiddenSize + 1) * sizeof(Weight));
		inputNeuron->Weights->Length = hiddenSize;

		for (int h = 0; h < hidden->Length; h++)
		{
			Neuron* pHiddenNeuron = &hidden->Items[h];
			Weight* pWeightI_H = &inputNeuron->Weights->Items[h];

			char hidBias = h == hidden->Length - 1;

			pHiddenNeuron->Value = hidBias ? 1 : 0;
			if (inpBias && hidBias) //no connection between biases
				continue;

			pWeightI_H->ConnectedNeuron = pHiddenNeuron;
			pWeightI_H->Value = ((double)rand() / (RAND_MAX));
			pHiddenNeuron->Weights = GlobalAlloc(0, sizeof(WeightsList));
						
			pHiddenNeuron->Weights->Items = GlobalAlloc(0, outputSize * sizeof(Weight));
			pHiddenNeuron->Weights->Length = outputSize;

			for (int o = 0; o < output->Length; o++)
			{
				Neuron* pOutputNeuron = &output->Items[o];
				Weight* pWeightH_O = &pHiddenNeuron->Weights->Items[o];

				pWeightH_O->ConnectedNeuron = pOutputNeuron;
				pWeightH_O->Value = ((double)rand() / (RAND_MAX));
				pOutputNeuron->Value = 0;
			}
		}
	}

}

void PrintAnn() {
	for (int l = 0; l < 3; l++)
	{
		NeuronsList* nl = &Ann.Layers[l];
		int length = nl->Length;
		printf("layer %d\n", l);
		for (int n = 0; n < length; n++)
		{
			Neuron* neuron = &nl->Items[n];
			printf("%f\n", neuron->Value);

			if (l < 2) //output has no weights
			{
				for (int w = 0; w < neuron->Weights->Length; w++)
				{
					printf("\t%f %f\n", neuron->Weights->Items[w].ConnectedNeuron->Value, neuron->Weights->Items[w].Value);
				}
			}
		}
		printf("\n");
	}
}

void PrintOutput() {
	NeuronsList* output = &Ann.Layers[2];
	for (size_t i = 0; i < output->Length; i++)
	{
		Neuron* neuron = &output->Items[i];
		printf("%f ", neuron->Value);
	}

	printf("   %f", Ann.TotalError);
	printf("\n");
}


void FreeANN() {	

	free(Ann.Layers[0].Items);
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

void Compute(double * data, int dataLength) {

	if (dataLength != Ann.Layers[0].Length - 1)
	{
		fprintf(stderr, "Length of input data is not same as Length of input layer.\n");
		exit(1000);
	}

	for (int i = 0; i < dataLength; i++)
		Ann.Layers[0].Items[i].Value = data[i];

	//Forward propagation
	for (int l = 0; l < 2; l++)
	{
		NeuronsList* layer = &Ann.Layers[l];
		for (int n = 0; n < layer->Length; n++)
		{
			Neuron * neuron = &layer->Items[n];
			for (int i = 0; i < neuron->Weights->Length; i++)
			{
				Weight* weight = &neuron->Weights->Items[i];
				weight->ConnectedNeuron->Value += (weight->Value * neuron->Value);
			}
		}

		// Applying aktivation function to all neurons in next layer
		NeuronsList* nextLayer = &Ann.Layers[l + 1];
		int neuronCount = nextLayer->Length - 1; //skipping bias neuron
		if (l + 1 == 2)
			neuronCount = nextLayer->Length; // last layer has no bias neuron

		for (int n = 0; n < neuronCount; n++)
		{
			Neuron * neuron = &nextLayer->Items[n];
			neuron->Value = LeakyReLU(neuron->Value / (layer->Length)); // dividing to prevent over flow.
			//neuron.Value = Sigmoid(neuron.Value / Layers[l].Count);
		}
	}
}

void BackProp(double * targets, int targLength) {

	if (targLength != Ann.Layers[2].Length)
	{
		fprintf(stderr, "Length of target data is not same as Length of output layer.\n");
		exit(1000);
	}

	Ann.TotalError = 0;
	
	NeuronsList* output = &Ann.Layers[2];;
	//backwards propagation
	for (int n = 0; n < output->Length; n++)
	{
		Neuron* neuron = &output->Items[n];
		neuron->Target = targets[n];
		neuron->Error = pow(neuron->Target - neuron->Value, 2) / 2;
		neuron->Delta = (neuron->Value - neuron->Target) * (neuron->Value > 0 ? 1 : 1 / (double)20);
		//neuron.Delta = (neuron->Value - neuron->Target) * (neuron->Value * (1 - neuron->Value));

		Ann.TotalError += neuron->Error;
	}

	NeuronsList* hidden = &Ann.Layers[1];;
	for (int i = 0; i < hidden->Length; i++)
	{
		// This can be done in parallell.
		Neuron* neuron = &hidden->Items[i];
		for (int w = 0; w < neuron->Weights->Length; w++)
		{
			Weight* weight = &neuron->Weights->Items[w];
			weight->Delta = neuron->Value * weight->ConnectedNeuron->Delta;
		}
	}
	
	//Input Layer
	NeuronsList* input = &Ann.Layers[0];
	for (int i = 0; i < input->Length; i++)
	{
		//Could be run in parallell
		Neuron* neuron = &input->Items[i];
		for (int w = 0; w < neuron->Weights->Length; w++)
		{
			Weight* weight = &neuron->Weights->Items[w];
			for (size_t c = 0; c < weight->ConnectedNeuron->Weights->Length; c++)
			{
				Weight * connectedWeight = &weight->ConnectedNeuron->Weights->Items[c];
				weight->Delta += connectedWeight->Value * connectedWeight->ConnectedNeuron->Delta;
			}
			double cv = weight->ConnectedNeuron->Value;
			weight->Delta *= cv > 0 ? 1 : 1 / (double)20;
			weight->Delta *= neuron->Value;
		}
	}

	//Parallel.ForEach(Layers[0], GetParallelOptions(), (neuron) = > {

	//	foreach(var weight in neuron.Weights)
	//	{
	//		foreach(var connectedWeight in weight.ConnectedNeuron.Weights)
	//			weight.Delta += connectedWeight.Value * connectedWeight.ConnectedNeuron.Delta;
	//		var cv = weight.ConnectedNeuron.Value;
	//		//weight.Delta *= (cv * (1 - cv));
	//		weight.Delta *= cv > 0 ? 1 : 1 / 20d;
	//		weight.Delta *= neuron.Value;
	//	}

	//});

	//All deltas are done. Now calculate new weights.
	for (int l = 0; l < 2; l++)
	{
		NeuronsList * layer = &Ann.Layers[l];
		for (size_t n = 0; n < layer->Length; n++)
		{
			Neuron* neuron = &layer->Items[n];
			for (size_t w = 0; w < neuron->Weights->Length; w++)
			{
				Weight* weight = &neuron->Weights->Items[w];
				weight->Value -= (weight->Delta * Ann.LearnRate);
			}
		}
	}
}