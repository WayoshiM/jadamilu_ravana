g++    -c -o spinhamiltonian.o spinhamiltonian.C
g++ spinhamiltonian.o -L/usr/lib64 -lm -lblas -L/home/gmccauley/jadamilu -ljadamilu -lmylapack -o spintest -lgfortran
