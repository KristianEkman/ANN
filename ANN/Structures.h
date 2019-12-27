#pragma once
struct Neuron;
struct Weight;

typedef struct Weight {
	float Value;
	float Delta;
	struct Neuron* ConnectedNeuron;
} Weight;

typedef struct Neuron {
	float Value;
	float Target;
	float Error;
	float Delta;
	struct WeightsList* Weights;
#ifdef _DEBUG
	char Tag[10];
#endif

} Neuron;

typedef struct WeightsList {
	unsigned int Length;
	struct Weight* Items;
} WeightsList;

typedef struct {
	unsigned int Length;
	struct Neuron* Items;
} NeuronsList;


typedef struct {
	NeuronsList Layers[3];
	float TotalError;
	const float LearnRate;
} ANN;

