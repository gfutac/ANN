#include <iostream>

using namespace std;

// abstract activation function
class ActivationFunc{
public:
	virtual double df(double input) = 0;
	virtual double f(double input) = 0;
	virtual ~ActivationFunc() {  }
};

// sigmoid
class Sigmoid : public ActivationFunc{
public:
	virtual double f(double input){
		return 1.0 / (1.0 + exp(-input));
	}

	virtual double df(double input){
		return input * (1.0 - input);		
	}

	virtual ~Sigmoid() { }
};

//hyperbolic tangens
class Tanh : public ActivationFunc{
public:
	virtual double f(double input){
		double ez = exp(input);
		double ezm = exp(-input);

		return (ez - ezm) / (ez + ezm);
	}

	virtual double df(double input){
		return 1.0 - input * input;
	}

	virtual ~Tanh() { }
};

// softplus
class Softplus : public ActivationFunc{
public:
	virtual double f(double input){
		return log10(1 + exp(input));
	}

	virtual double df(double input){
		return 1.0 / (1 + exp(-input));
	}

	virtual ~Softplus() { }
};
