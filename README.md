Public
======


Aritifical neural network. Only MLP trained with basic backpropagation is implemented for now. 
Logistic and hyperbolic tangens activation functions are supported.

Boost and OpenCV are required for succesful compilation. OpenCV is used for BLAS, simply because I'm used to it.

For now, problem is read from file.
First line in input file contains single number - problem dimensionality. Eeach of the following lines
contains samples and expected outputs, separated with space.
eq, if dimensionality is 2,
1 0 1 represetnes sample [1, 0], and output [1]

