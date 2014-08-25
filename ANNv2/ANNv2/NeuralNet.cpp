#include <iostream>
#include <vector>
#include <random>
#include <ctime>

#include <boost/foreach.hpp>

#include "NeuralNet.h"
#include "TrainingAlgorithm.h"
#include "BackPropagation.h"

using namespace std;

#define foreach_ BOOST_FOREACH

NeuralNet::NeuralNet(const Problem &problem, const vector<int> args, double epsilon, double regularizationFactor, std::unique_ptr<ActivationFunc> af){
	this->epsilon = epsilon;
	this->problem = problem;
	this->af = std::move(af);
	this->regularizationFactor = regularizationFactor;

	std::default_random_engine generator((unsigned long)time(0));
	std::normal_distribution<double> distribution(0, 0.5);

	for (int i = 0; i < args.size() - 1; ++i){
		int rows = args[i] + 1;
		int cols = args[i + 1];

		Mat m(rows, cols, CV_64F);
		int c = 0;
		std::transform((double *)(m.data), (double *)(m.data) + rows * cols, (double *)(m.data), [&](const double &v){ 
			int idx = &v - &((double *)(m.data))[0];
			double ret = idx < (rows - 1) * cols ? distribution(generator) : 1; // set biases to 1
			return ret;
		});

		this->w.push_back(m);
	}
}

NeuralNet::NeuralNet(const Problem &problem, vector<int> args) :
NeuralNet(problem, args, 0.0005, 1.0, std::unique_ptr<ActivationFunc>(new Sigmoid())) { }

Mat NeuralNet::predict(Mat &sample){
	Mat x = sample;

	for (int i = 0; i < this->w.size(); ++i){
		Mat activations = x * this->w[i];

		int dim = activations.rows * activations.cols;
		std::transform((double *)activations.data, (double *)activations.data + dim, (double *)activations.data, [&](const double &value) { return this->af->f(value); });
		hconcat(activations, Mat::ones(1, 1, CV_64F), activations);

		x = activations;
	}

	return x.colRange(0, x.cols - 1);
}

NeuralNet::~NeuralNet(){
	//delete this->af;
}
