Aritifical neural network. Only MLP trained with basic backpropagation is implemented for now. 
Logistic and hyperbolic tangens activation functions are supported.

Boost and OpenCV are required for succesful compilation. OpenCV is used for BLAS, simply because I'm used to it :)
I'm planning to replace OpenCV with some other, more lightweight LAPACK library.

I will try to implement autoencoder next.


For now, problem is read from file with following structure: 

2
0 0 0
1 1 0
0 1 1
1 0 1

2 in first line sample dimensionality. As you can see, this is XOR problem.
I'm planning to add layer sizes in same file too (currently they are hardcoded, this is still under heavy development :))
