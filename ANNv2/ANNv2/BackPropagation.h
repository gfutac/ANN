#ifndef backpropagation_h
#define backpropagation_h

#include <iostream>
#include <vector>
#include <list>

#include "TrainingAlgorithm.h"

using namespace std;

class BackPropagation : public TrainingAlgorithm {
private:
	vector<Mat> activations;
	vector<Mat> derivations;
	list<Mat> deltas;

	void forwardpass(NeuralNet &net, Mat sample);
	void backwardpass(NeuralNet &net, Mat y);

public:
	virtual void train(NeuralNet &net, double learningRate);
};

#endif backpropagation_h