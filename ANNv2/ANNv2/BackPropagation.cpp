#include <iostream>
#include <boost\foreach.hpp>

#include "BackPropagation.h"
#include "NeuralNet.h"

using namespace std;

#define foreach_ BOOST_FOREACH

void BackPropagation::train(NeuralNet &net, double learningRate){
	Mat samples = net.getProblem().samples;
	Mat outputs = net.getProblem().outputs;


	int iter = 0;
	do {
		net.error = 0;
		for (int i = 0; i < samples.rows; ++i){
			Mat x = samples.row(i);
			Mat y = outputs.row(i);

			this->forwardpass(net, x);
			this->backwardpass(net, y);

			if (iter % 1000 == 0){
				cout << "input " << x << " class " << y << " predicted " << net.h << endl;
			}
		}

		if (iter % 1000 == 0) {
			cout << endl;
		}

		iter++;
	} while (net.error > net.epsilon);

	cout << endl << endl << "iterations " << iter << endl;
	cout << "error " << net.error << endl << endl;

}

void BackPropagation::forwardpass(NeuralNet &net, Mat sample){
	this->activations.clear();
	this->derivations.clear();
	this->deltas.clear();

	// forward pass
	Mat x = sample;

	this->activations.push_back(x.colRange(0, x.cols - 1));
	for (int i = 0; i < net.w.size(); ++i){
		Mat sums = x * net.w[i];

		int dim = sums.rows * sums.cols;
		std::transform((double *)sums.data, (double *)sums.data + dim, (double *)sums.data, [&](const double &value) { return net.af->f(value); });
		hconcat(sums, Mat::ones(1, 1, CV_64F), sums);
		Mat der = sums.clone();
		std::transform((double *)der.data, (double *)der.data + dim, (double *)der.data, [&](const double &value) {	return net.af->df(value); });

		this->activations.push_back(sums.colRange(0, sums.cols - 1));
		this->derivations.push_back(der);

		x = sums;
	}
}

void BackPropagation::backwardpass(NeuralNet &net, Mat y){
	// output layer deltas
	Mat err = y - this->activations.back();
	Mat tmpErr = err * err.t();
	double tmp = sum(tmpErr)[0] / 2.0;
	net.error += tmp;

	Mat delta = -(err).mul(derivations.back());
	net.h = activations.back();
	activations.pop_back();
	delta = delta.colRange(0, delta.cols - 1);
	deltas.push_front(delta);

	// hidden layers deltas
	for (int i = net.w.size() - 1; i > 0; --i){
		Mat W = net.w[i].rowRange(0, net.w[i].rows - 1);
		Mat d = derivations[i - 1].colRange(0, derivations[i - 1].cols - 1).t();

		delta = (W * delta).mul(d);
		deltas.push_front(delta);
	}

	int i = 0;
	foreach_(Mat delta, deltas) {
		Mat d = 0.5 * delta;
		Mat tmpActivation = d * activations[i];
		hconcat(tmpActivation, d, tmpActivation);
		net.w[i++] -= tmpActivation.t();
	}
}
