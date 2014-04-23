Aritifical neural network. Only MLP trained with basic backpropagation is implemented for now. 
Logistic and hyperbolic tangens activation functions are supported.

Boost and OpenCV are required for succesful compilation. OpenCV is used for BLAS, simply because I'm used to it :)
I'm planning to replace OpenCV with some other, more lightweight LAPACK library.

I will try to implement autoencoder next.


For now, problem is read from file 
First line in input file contains single number - problem dimensionality. Eeach of the following lines
contains samples and expected outputs, separated with space.
eq, if dimensionality is 2,
1 0 1 represetnes sample [1, 0], and output [1]


I'm planning to add layer sizes in same file too (currently they are hardcoded, this is still under heavy development :))
