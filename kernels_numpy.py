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
from numba import jit, prange, get_num_threads
import scipy

# note: we have not implemented SELL-C-sigma spmv here
from kernels_cpu import memory_benchmarks, sell_spmv

def copy_vector(x):
    return x.copy()

def copy_csr_arrays(Adata, Aindptr, Aindices):
    return Adata.copy(), Aindices.copy(), Aindptr.copy()

def init(x, val):
    x[:] = val

def axpby(a,x,b,y):
    y[:] = a*x + b*y

def dot(x,y):
    return np.dot(x,y)

def csr_spmv(valA,rptrA,colA, x, y):
    nrows = len(y)
    ncols = len(x)
    A = scipy.sparse.csr_matrix((valA, colA, rptrA), shape=[nrows, ncols])
    y[:] = A*x
