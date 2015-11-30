LD_LIBRARY_PATH=/home/gmccauley/jadamilu
export LD_LIBRARY_PATH
gcc -c -D_REENTRANT -fPIC \
    -I/home/duse/downloads/jdk1.7.0_07/include \
    -I/home/duse/downloads/jdk1.7.0_07/include/linux -c jadamilu.c
gcc -shared jadamilu.o /usr/lib64/libblas.so libjadamilu.a /usr/lib64/liblapack.so.3.0.3 -o libjadamilu.so
