#ifndef trainingalgorithm_h
#define trainingalgorithm_h

#include <iostream>
#include <opencv2\core\core.hpp>

using namespace cv;
using namespace std;

class NeuralNet;

class TrainingAlgorithm {
public:
	virtual void train(NeuralNet &net, double learningRate) = 0;

	double learningRate;
	double weightDecay = 0;
};

#endif trainingalgorithm_h