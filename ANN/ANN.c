#include "Structures.h"
#include "ANN.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void SetInputTag(Neuron* n, char* layer, int index) {
	char buffer[10];
	snprintf(buffer, 10, "%s%d", layer, index);
	strcpy_s(n->Tag, 10, buffer);
}

void NewAnn(int inputSize, int hiddenSize, int outputSize) {
	NeuronsList* input = &Ann.Layers[0];
	input->Items = malloc(inputSize + 1 * sizeof(Neuron));//allocating for bias
	input->Length = inputSize;

	NeuronsList* hidden = &Ann.Layers[1];
	hidden->Items = malloc((hiddenSize + 1) * sizeof(Neuron));//allocating for bias
	hidden->Length = hiddenSize;

	NeuronsList* output = &Ann.Layers[2];
	output->Items = malloc(outputSize * sizeof(Neuron));
	output->Length = outputSize;

	// Dont forget the bias
	Neuron* inputNeuron = input->Items;
	for (int i = 0; i < input->Length + 1; i++)
	{
		char inpBias = i == input->Length;
		inputNeuron->Value = inpBias ? 1 : 0;

		WheightsList* p_weightsListI_H = malloc(sizeof(WheightsList));
		p_weightsListI_H->Items = malloc((hidden->Length + 1) * sizeof(Wheight));
		p_weightsListI_H->Length = hidden->Length;
		inputNeuron->Wheights = p_weightsListI_H;

		SetInputTag(inputNeuron, inpBias ? "B" : "I", i);

		Neuron* pHiddenNeuron = hidden->Items;
		Wheight* pWeightI_H = p_weightsListI_H->Items;
		for (int h = 0; h < hidden->Length + 1; h++)
		{
			char hidBias = h == hidden->Length;

			pHiddenNeuron->Value = hidBias ? 1 : 0;
			if (inpBias && hidBias) //no connection between biases
				continue;

			pWeightI_H->ConnectedNeuron = pHiddenNeuron;
			pWeightI_H->Value = ((double)rand() / (RAND_MAX));
			WheightsList* p_weightsListH_O = malloc(sizeof(WheightsList));
			p_weightsListH_O->Items = malloc(output->Length * sizeof(Wheight));
			p_weightsListH_O->Length = output->Length;
			pHiddenNeuron->Wheights = p_weightsListH_O;

			SetInputTag(pHiddenNeuron, hidBias ? "B" : "H", h);

			Neuron* pOutputNeuron = output->Items;
			Wheight* pWeightH_O = p_weightsListH_O->Items;
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
		NeuronsList * nl = &Ann.Layers[l];
		int length = nl->Length + 1;
		if (l == 2)
			length = nl->Length; //for all layers but last also print bias

		for (int j = 0; j < length; j++)
		{
			Neuron * neuron = &nl->Items[j];
			printf("%s (%f)\n", neuron->Tag, neuron->Value);

			if (l < 2) //output has no weights
				for (int w = 0; w < neuron->Wheights->Length; w++)
				{
					printf("\t%s %f\n", neuron->Wheights->Items[w].ConnectedNeuron->Tag, neuron->Wheights->Items[w].Value);
				}
		}
		printf("\n");
	}
}