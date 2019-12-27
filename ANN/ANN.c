#include "Structures.h"
#include "ANN.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Windows.h>
#include <math.h>



void SetInputTag(Neuron* n, char* layer, int index) {
	char buffer[10];
	snprintf(buffer, 10, "%s%d\0", layer, index);
#ifdef _DEBUG
	strcpy_s(n->Tag, 10, buffer);
#endif // DEBUG

}

void NewAnn(int inputSize, int hiddenSize, int outputSize) {
	NeuronsList* input = &Ann.Layers[0];
	input->Items = GlobalAlloc(0, (inputSize + 1) * sizeof(Neuron));//allocating for bias
	input->Length = inputSize;

	NeuronsList* hidden = &Ann.Layers[1];
	hidden->Items = GlobalAlloc(0, (hiddenSize + 1) * sizeof(Neuron));//allocating for bias
	hidden->Length = hiddenSize;

	NeuronsList* output = &Ann.Layers[2];
	output->Items = GlobalAlloc(0, outputSize * sizeof(Neuron));
	output->Length = outputSize;

	Neuron* inputNeuron = input->Items;
	for (int i = 0; i < input->Length + 1; i++)
	{
		char inpBias = i == input->Length;
		inputNeuron->Value = inpBias ? 1 : 0;

		WeightsList* p_weightsListI_H = GlobalAlloc(0, sizeof(WeightsList));
		p_weightsListI_H->Items = GlobalAlloc(0, (hidden->Length + 1) * sizeof(Weight));
		p_weightsListI_H->Length = hidden->Length;
		inputNeuron->Weights = p_weightsListI_H;
#ifdef _DEBUG
		SetInputTag(inputNeuron, inpBias ? "B\0" : "I\0", i);
#endif


		Neuron* pHiddenNeuron = hidden->Items;
		Weight* pWeightI_H = p_weightsListI_H->Items;
		for (int h = 0; h < hidden->Length + 1; h++)
		{
			char hidBias = h == hidden->Length;

			pHiddenNeuron->Value = hidBias ? 1 : 0;
			if (inpBias && hidBias) //no connection between biases
				continue;

			pWeightI_H->ConnectedNeuron = pHiddenNeuron;
			pWeightI_H->Value = ((float)rand() / (RAND_MAX));
			WeightsList* p_weightsListH_O = GlobalAlloc(0, sizeof(WeightsList));
			p_weightsListH_O->Items = GlobalAlloc(0, output->Length * sizeof(Weight));
			p_weightsListH_O->Length = output->Length;
			pHiddenNeuron->Weights = p_weightsListH_O;
#ifdef _DEBUG
			SetInputTag(pHiddenNeuron, hidBias ? "B\0" : "H\0", h);
#endif
			Neuron* pOutputNeuron = output->Items;
			Weight* pWeightH_O = p_weightsListH_O->Items;
			for (int o = 0; o < output->Length; o++)
			{
				pWeightH_O->ConnectedNeuron = pOutputNeuron;
				pWeightH_O->Value = ((float)rand() / (RAND_MAX));
				pOutputNeuron->Value = 0;
#ifdef _DEBUG
				SetInputTag(pOutputNeuron, "O", o);
#endif
				pOutputNeuron++;
				pWeightH_O++;
			}
			pWeightI_H++;
			pHiddenNeuron++;
		}
		inputNeuron++;
	}

}

#ifdef _DEBUG
void PrintAnn() {
	for (int l = 0; l < 3; l++)
	{
		NeuronsList* nl = &Ann.Layers[l];
		int length = nl->Length + 1;
		if (l == 2)
			length = nl->Length; //for all layers but last also print bias

		for (int j = 0; j < length; j++)
		{
			Neuron* neuron = &nl->Items[j];
			printf("%s (%f)\n", neuron->Tag, neuron->Value);

			if (l < 2) //output has no weights
				for (int w = 0; w < neuron->Weights->Length; w++)
				{
					printf("\t%s %f\n", neuron->Weights->Items[w].ConnectedNeuron->Tag, neuron->Weights->Items[w].Value);
				}
		}
		printf("\n");
	}
}
#endif


void FreeANN() {	

	GlobalFree(Ann.Layers[0].Items);
}

double LeakyReLU(float x)
{
	if (x >= 0)
		return x;
	else
		return x / 20;
}

double Sigmoid(double x)
{
	float s = 1 / (1 + exp(-x));
	return s;
}

void Compute(float * data, int dataLength) {

	for (int i = 0; i < dataLength; i++)
		Ann.Layers[0].Items[i].Value = data[i];

	//Forward propagation
	for (int l = 0; l < 2; l++)
	{
		for (int n = 0; n <  Ann.Layers[l].Length; n++)
		{
			Neuron * neuron = &Ann.Layers[l].Items[n];
			for (size_t i = 0; i < neuron->Weights->Length; i++)
			{
				Weight* weight = &neuron->Weights->Items[i];
				weight->ConnectedNeuron->Value += (weight->Value * neuron->Value);
			}
		}

		int neuronCount = Ann.Layers[l + 1].Length;
		//if (l + 1 < Layers.Count() - 1)
		//	neuronCount--; //skipping bias
		for (int n = 0; n < neuronCount; n++) //next layer
		{
			Neuron * neuron = &Ann.Layers[l + 1].Items[n];
			neuron->Value = LeakyReLU(neuron->Value / (Ann.Layers[l+1].Length + 1));
			//neuron.Value = Sigmoid(neuron.Value / Layers[l].Count);
		}
	}
}