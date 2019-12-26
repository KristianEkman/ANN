#pragma once
struct Neuron;
struct Weight;

typedef struct Weight {
	double Value;
	double Delta;
	struct Neuron* ConnectedNeuron;
} Weight;

typedef struct Neuron {
	double Value;
	double Target;
	double Error;
	double Delta;
	struct WeightsList* Weights;
	//char Tag[10];
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
	double TotalError;
	const double LearnRate;
} ANN;

