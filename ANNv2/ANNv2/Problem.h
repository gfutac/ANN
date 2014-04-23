#ifndef problem_h
#define problem_h


#include <opencv2\core\core.hpp>

using namespace cv;
using namespace std;

class Problem {
public:
	Problem();
	Problem(const string filename, const string delimiter, bool extend = true);
	~Problem();
	Mat samples;
	Mat outputs;
	int problemSize;
	vector<int> hiddenLayers;
};


#endif problem_h
