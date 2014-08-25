#ifndef neuralnet_h
#define neuralnet_h

#include <iostream>
#include <vector>
#include <list>
#include <memory>

#include <opencv2\core\core.hpp>

#include "ActivationFunc.h"
#include "Problem.h"

using namespace cv;
using namespace std;

class NeuralNet {
private:
	Problem problem;
	
public:
	NeuralNet(const Problem &problem, vector<int> args);
	NeuralNet(const Problem &problem, vector<int> args, double epsilon, double regularizationFactor, std::unique_ptr<ActivationFunc> af);
	Problem getProblem() { return this->problem; }
	Mat predict(Mat &sample);
	~NeuralNet();

	vector<Mat> w;	
	Mat h;
	std::unique_ptr<ActivationFunc> af;

	double epsilon;
	double error;
	double regularizationFactor;
};

#endif neuralnet_h