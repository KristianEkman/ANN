#pragma once
struct Neuron;
struct Wheight;

typedef struct Wheight {
	double Value;
	double Delta;
	struct Neuron* ConnectedNeuron;
} Wheight;

typedef struct Neuron {
	double Value;
	double Target;
	double Error;
	double Delta;
	struct WheightsList* Wheights;
	char Tag[10];
} Neuron;

typedef struct WheightsList {
	unsigned int Length;
	struct Wheight* Items;
} WheightsList;

typedef struct {
	unsigned int Length;
	struct Neuron* Items;
} NeuronsList;


typedef struct {
	NeuronsList Layers[3];
	double TotalError;
	const double LearnRate;
} ANN;

