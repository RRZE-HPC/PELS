# If you want to run with C/OpenMP backend (-c_kernels or -use_RACE),
# you need to specify a compiler and flags here.
# Note that it can be challenging to pick a good combination with Python
# as Numba uses LLVM and this may interfere with, e.g., GCC multi-threading.
# In our experience, using the LLVM-based Intel compiler (icx) gives best results,
# in combination with explicitly disabling the garbage collector (gc.disable() in Python).

## GCC
#CC=gcc
#CFLAGS=-fPIC -O3 -fopenmp -march=native
#CXX=g++
#CXXFLAGS=${CFLAGS}

## Intel LLVM-based compiler
CC=icx
CFLAGS=-fPIC -O3 -qopenmp -xHOST -fimf-use-svml=true #-Xclang -target-feature -Xclang +prefer-no-gather
CXX=icpx
CXXFLAGS=${CFLAGS}

## Intel legacy compilers
#CC=icc
#CFLAGS=-fPIC -O3 -qopenmp -xHOST
#CXX=icpc
#CXXFLAGS=${CFLAGS}

# In order to enable RACE cache-blocking, you also have to set these variables:
RACE_INCDIR=${HOME}/local/include
RACE_LIBDIR=${HOME}/local/lib/RACE

