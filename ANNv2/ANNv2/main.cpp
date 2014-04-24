#include <string>
#include <set>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <ctime>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include <opencv2\core\core.hpp>

using namespace cv;

#include "Problem.h"
#include "NeuralNet.h"
#include "TrainingAlgorithm.h"
#include "BackPropagation.h"

#define foreach_ BOOST_FOREACH

using namespace boost;
using namespace std;

template<typename... params>
vector<int> layers(params... pr){
	vector<int> p = { pr... };
	return p;
}

int main(int argc, char **argv){

	const string filename = "D:\\test.txt";

	Problem problem(filename, " ");
	NeuralNet nn(problem, layers(2, 2, 1));
	nn.epsilon = 0.0001;
	nn.regularizationFactor = 1;
	nn.af = new Tanh();

	// train ann with backpropagation
	TrainingAlgorithm *ta = new BackPropagation();
	double learningRate = 0.5;
	ta->train(nn, learningRate);


	delete ta;

	system("pause");
	return EXIT_SUCCESS;
}