#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import numpy as np
from ctypes import *
from numpy.ctypeslib import as_ctypes
import os

# compile the C code into a shared library

# CLANG -- doesn't vectorize well
#os.system('clang -frtlib-add-rpath -fPIC -O3 -fopenmp -fvectorize -ffast-math -march=native -shared -o vector_ops.so vector_ops.c')

# Intel LLVM-based (icx)
#os.system('icx -fPIC -O3 -qopenmp -march=native -fimf-use-svml=true -shared -o vector_ops.so vector_ops.c')

# GCC
os.system('gcc -fPIC -O3 -fopenmp -ffast-math -march=native -shared -o vector_ops.so vector_ops.c')

# import the C library -> c_functions object
so_file = "./vector_ops.so"
c_functions = CDLL(so_file)

# convenient alias
c_double_p = POINTER(c_double)

# need to explicitly set the argument types to avoid runtime errors for some reason...
c_functions.axpy.argtypes = [c_size_t, c_double, c_double_p, c_double_p, c_double_p]
c_functions.dot.argtypes = [c_size_t, c_double_p, c_double_p]
c_functions.foo.argtypes = [c_size_t, c_double, c_double_p, c_double_p, c_double_p]


def axpy(a,x,y,z):
    N=x.shape[0]
    c_functions.axpy(N,a, as_ctypes(x), as_ctypes(y), as_ctypes(z))
    return z

def dot(x,y):
    N=x.shape[0]
    s=0.0
    s=c_functions.dot(N,as_ctypes(x),as_ctypes(y))
    return s

def foo(a,x,y,z):
    N=x.shape[0]
    c_functions.foo(N,a, as_ctypes(x),as_ctypes(y), as_ctypes(z))
    return z

if __name__=='__main__':
    N = 100
    x = np.random.rand(N)
    y = np.random.rand(N)
    z = np.zeros(N)
    z = axpy(3.14,x,y,z)
    s = dot(x,y)
    z = foo(3.14,x,y,z)
