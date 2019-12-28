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
} Neuron;

typedef struct WeightsList {
	unsigned int Length;
	struct Weight* Items;
} WeightsList;

typedef struct NeuronsList {
	unsigned int Length;
	struct Neuron* Items;
} NeuronsList;


typedef struct {
	NeuronsList Layers[3];
	double TotalError;
	double LearnRate;
} ANN;

