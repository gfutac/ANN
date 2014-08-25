#include <string>
#include <set>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <ctime>
#include <memory>

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

	auto ta = unique_ptr<TrainingAlgorithm>(new BackPropagation);

	Problem problem(filename, " ");
	NeuralNet nn(problem, layers(2, 3, 1), 0.0001, 1, unique_ptr<ActivationFunc>(new Tanh));

	double learningRate = 0.5;
	ta->weightDecay = 1;
	ta->train(nn, learningRate);

	system("pause");
	return EXIT_SUCCESS;
}