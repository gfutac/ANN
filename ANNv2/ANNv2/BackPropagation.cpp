#include <iostream>
#include <boost\foreach.hpp>

#include "BackPropagation.h"
#include "NeuralNet.h"

using namespace std;

#define foreach_ BOOST_FOREACH

void BackPropagation::train(NeuralNet &net, double learningRate){
	Mat samples = net.getProblem().samples;
	Mat outputs = net.getProblem().outputs;
	this->learningRate = learningRate;

	this->w_count = 0;
	foreach_(Mat wi, net.w) {
		this->w_count += (wi.rows) * wi.cols;
	}

	int iter = 0;
	do {
		net.error = 0;
		for (int i = 0; i < samples.rows; ++i){
			Mat x = samples.row(i);
			Mat y = outputs.row(i);

			this->forwardpass(net, x);
			this->backwardpass(net, y);

			// regularization
			net.error = net.regularizationFactor * net.error + (1 - net.regularizationFactor) * (this->w_squaredsum / (double)this->w_count);
			
			if (iter % 1000 == 0){
				cout << "input " << x << " class " << y << " predicted " << net.h << endl;				
			}
		}

		if (iter % 1000 == 0) {
			cout << "iter: " << iter << "\t" << "error: " << net.error << endl;
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
	Mat sums;

	this->activations.push_back(x.colRange(0, x.cols - 1));
	for (int i = 0; i < net.w.size(); ++i) {			
		// sum = W(transposed) * x		
		sums = x * net.w[i];
		int dim = sums.rows * sums.cols;

		// calculate activations - transform matrix, calculate activation f(Wt * x)
		std::transform((double *)sums.data, (double *)sums.data + dim, (double *)sums.data, [&](const double &value) { return net.af->f(value); });
		
		// calculate derivations
		Mat der = sums.clone();
		std::transform((double *)der.data, (double *)der.data + dim, (double *)der.data, [&](const double &value) {	return net.af->df(value); });

		// store activations and derivations
		this->activations.push_back(sums);
		this->derivations.push_back(der);

		// activations are input to next layer
		x = sums;
		hconcat(x, Mat::ones(1, 1, CV_64F), x);
	}

	net.h = sums;
	activations.pop_back();
}

void BackPropagation::backwardpass(NeuralNet &net, Mat y){
	// calculate mean squared error ( 1/2 * sum (err*err)
	Mat err = y - net.h;
	Mat tmpErr = err * err.t();
	double tmp = sum(tmpErr)[0] / net.getProblem().samples.rows;
	net.error += tmp;

	// output layer deltas -> -(y - h) .* f'(x)
	Mat delta = -(err).mul(derivations.back());
	deltas.push_front(delta);

	// hidden layers deltas
	for (int i = net.w.size() - 1; i > 0; --i){
		Mat W = net.w[i].rowRange(0, net.w[i].rows - 1);
		Mat d = derivations[i - 1].t();

		delta = (W * delta).mul(d);
		deltas.push_front(delta);
	}

	int i = 0;
	this->w_squaredsum = 0;
	double m = net.getProblem().samples.rows;

	foreach_(Mat delta, deltas) {
		Mat d = this->learningRate * delta;
		Mat dw = d * activations[i];

		//cout << i << endl;
		//cout << dw.size() << endl;
		//cout << net.w[i].colRange(0, net.w[i].cols - 1).size() << endl;

		//dw = dw + this->learningRate * net.w[i].colRange(0, net.w[i].cols - 1) * 0.005;
		hconcat(dw, d, dw);
		
		net.w[i] -=  dw.t();

		// needed for regularization
		Mat tmpW = net.w[i].mul(net.w[i]);
		this->w_squaredsum += sum(tmpW)[0];

		++i;
	}
}
