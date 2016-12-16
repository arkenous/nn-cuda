# nn-cuda
SdA - MLP neural network program written in C++ and CUDA ([Thrust](https://github.com/thrust/thrust)).

###Compile
```
$ nvcc -std=c++11 -O3 -I. *.cpp *.cu -o nn-cuda
```

###Run
```
$ ./nn-cuda
```