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
	nn.epsilon = 0.00001;

	nn.af = new Tanh();
	TrainingAlgorithm *ta = new BackPropagation();

	// train ann with backpropagation
	ta->train(nn, 0.5);

	double samples[] = { 0.1, 0.3, 1,
		1.2, 0.9, 1,
		0.4, 1.0, 1 };
	Mat ss(3, 3, CV_64F, samples);
	for (int i = 0; i < ss.rows; ++i){
		Mat prediction = nn.predict(ss.row(i));
		cout << ss.row(i) << " " << prediction << endl;
	}


	delete ta;

	system("pause");
	return EXIT_SUCCESS;
}