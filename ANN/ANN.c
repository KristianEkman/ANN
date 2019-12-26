#include "Structures.h"
#include "ANN.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <Windows.h>


void SetInputTag(Neuron* n, char* layer, int index) {
	char buffer[10];
	snprintf(buffer, 10, "%s%d\0", layer, index);
	//strcpy_s(n->Tag, 10, buffer);
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

		SetInputTag(inputNeuron, inpBias ? "B\0" : "I\0", i);

		Neuron* pHiddenNeuron = hidden->Items;
		Weight* pWeightI_H = p_weightsListI_H->Items;
		for (int h = 0; h < hidden->Length + 1; h++)
		{
			char hidBias = h == hidden->Length;

			pHiddenNeuron->Value = hidBias ? 1 : 0;
			if (inpBias && hidBias) //no connection between biases
				continue;

			pWeightI_H->ConnectedNeuron = pHiddenNeuron;
			pWeightI_H->Value = ((double)rand() / (RAND_MAX));
			WeightsList* p_weightsListH_O = GlobalAlloc(0, sizeof(WeightsList));
			p_weightsListH_O->Items = GlobalAlloc(0, output->Length * sizeof(Weight));
			p_weightsListH_O->Length = output->Length;
			pHiddenNeuron->Weights = p_weightsListH_O;

			SetInputTag(pHiddenNeuron, hidBias ? "B\0" : "H\0", h);

			Neuron* pOutputNeuron = output->Items;
			Weight* pWeightH_O = p_weightsListH_O->Items;
			for (int o = 0; o < output->Length; o++)
			{
				pWeightH_O->ConnectedNeuron = pOutputNeuron;
				pWeightH_O->Value = ((double)rand() / (RAND_MAX));
				pOutputNeuron->Value = 0;

				SetInputTag(pOutputNeuron, "O", o);

				pOutputNeuron++;
				pWeightH_O++;
			}
			pWeightI_H++;
			pHiddenNeuron++;
		}
		pHiddenNeuron->Value = 0;

		inputNeuron++;
	}

}

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
			//printf("%s (%f)\n", neuron->Tag, neuron->Value);

			if (l < 2) //output has no weights
				for (int w = 0; w < neuron->Weights->Length; w++)
				{
					//printf("\t%s %f\n", neuron->Weights->Items[w].ConnectedNeuron->Tag, neuron->Weights->Items[w].Value);
				}
		}
		printf("\n");
	}
}

void FreeANN() {	

	GlobalFree(Ann.Layers[0].Items);
}