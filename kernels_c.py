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
from numpy.ctypeslib import as_ctypes, as_array
import os

# some stuff is missing here up to now:
from kernels_cpu import memory_benchmarks, sell_spmv

# compile the C code into a shared library
os.system("make -j")


# import the C library -> c_functions object
so_file = "./libkernels.so"
c_functions = CDLL(so_file)

# convenient alias
c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

# need to explicitly set the argument types to avoid runtime errors for some reason...
c_functions.csr_spmv.argtypes = [c_size_t, c_double_p, c_int_p, c_int_p, c_double_p, c_double_p]
c_functions.axpby.argtypes = [c_size_t, c_double, c_double_p, c_double, c_double_p]
c_functions.vscale.argtypes = [c_size_t, c_double_p, c_double_p, c_double_p]
c_functions.dot.argtypes = [c_size_t, c_double_p, c_double_p]
c_functions.dot.restype = c_double
c_functions.init.argtypes = [c_size_t, c_double_p, c_double]
c_functions.copy_vector.argtypes = [c_size_t, c_double_p, c_double_p]
c_functions.copy_csr_arrays.argtypes = [c_size_t, c_double_p, c_int_p, c_int_p, c_double_p, c_int_p, c_int_p]


def csr_spmv(valA, rptrA, colA, x, y):
    N=y.shape[0]
    c_functions.csr_spmv(N, as_ctypes(valA), as_ctypes(rptrA), as_ctypes(colA), as_ctypes(x), as_ctypes(y))

def init(x, val):
    N = x.size
    c_functions.init(N, as_ctypes(x), val)

def copy_vector(x):
    N = x.size
    y = np.empty_like(x)
    c_functions.copy_vector(N, as_ctypes(x), as_ctypes(y))
    return y

def copy_csr_arrays(Adata, Aindptr, Aindices):
    data = np.empty_like(Adata)
    indices = np.empty_like(Aindices)
    indptr = np.empty_like(Aindptr)
    nrows = len(indptr)-1
    nnz   = len(Adata)
    c_functions.copy_csr_arrays(nrows,as_ctypes(Adata),as_ctypes(Aindptr),as_ctypes(Aindices),
                as_ctypes(data),as_ctypes(indptr),as_ctypes(indices))
    return data, indices, indptr

def axpby(a,x,b,y):
    N = x.size
    c_functions.axpby(N, a, as_ctypes(x), b, as_ctypes(y))

def dot(x,y):
    N = x.size
    s = c_functions.dot(N, as_ctypes(x), as_ctypes(y))
    return s

def vscale(v, x, y):
    '''
    Vector scaling y[i] = v[i]*x[i]
    '''
    N = x.size
    c_functions.vscale(N, as_ctypes(v), as_ctypes(x), as_ctypes(y))

def multiple_axpbys(a, x, b, y, ntimes):
    for it in range(ntimes):
        axpby(a,x,b,y)


