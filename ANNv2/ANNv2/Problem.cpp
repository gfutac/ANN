#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>

#include <boost/algorithm/string.hpp>
#include <opencv2\core\core.hpp>

#include "Problem.h"

using namespace cv;
using namespace std;

Problem::Problem() { }

Problem::Problem(const string filename, const string delimiter, bool extend){
	vector<string> lines;

	ifstream f(filename);
	if (!f.is_open()){
		cerr << "Error reading input file" << endl;
		exit(1);
	}

	cout << "Reading file..." << endl;
	
	string line;
	getline(f, line);
	int sampleSize = stoi(line);

	while (getline(f, line)){
		lines.push_back(line);
	}

	this->problemSize = lines.size();
	cout << "Parsing lines..." << endl;

	double *tmpProblem = 0;
	int w = 0, h = 0;

	for (int i = 0; i < this->problemSize; ++i){
		vector<string> tokens;
		string line = lines[i];
		boost::split(tokens, line, boost::is_any_of(delimiter), boost::token_compress_on);

		if (tmpProblem == 0){
			w = tokens.size();
			h = this->problemSize;
			tmpProblem = new double[h * w];
		}

		vector<double> doubles(w);
		transform(tokens.begin(), tokens.end(), doubles.begin(), [](string const& val) { return stod(val); });		
		memcpy(tmpProblem + i * w, &doubles[0], sizeof(double) * w);
	}

	Mat m(h, w, CV_64F, tmpProblem);
	this->samples = m.colRange(0, sampleSize).clone();
	this->outputs = m.colRange(sampleSize, w).clone();
	
	if (extend){
		hconcat(this->samples, Mat::ones(h, 1, CV_64F), this->samples);
	}
}

Problem::~Problem(){
}