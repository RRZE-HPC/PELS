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
import scipy
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve_triangular
from numba import cuda

def as_cupy(A, as_csc=False):
    '''
    If A is a vector or CSR matrix as used in the numba cuda_kernels,
    return a correpsonding "CuPy view" of the GPU arrays.

    If A is a CSR matrix and the flagas_csc=True is given, no data is moved, but the matrix
    is returned in compressed column instead of compressed row storage.
    That is, the returned object actually represents A.T, but if A is symmetric, that's the same thing.
    '''
    if type(A) == scipy.sparse.csc_matrix:
        as_csc = True
    if type(A) == scipy.sparse.csr_matrix or type(A) == scipy.sparse.csc_matrix:
        cp_data = cp.asarray(A.cu_data)
        cp_indptr = cp.asarray(A.cu_indptr)
        cp_indices = cp.asarray(A.cu_indices)
        if as_csc==True:
            return csp.csc_matrix((cp_data, cp_indices, cp_indptr))
        else:
            return csp.csr_matrix((cp_data, cp_indices, cp_indptr))
    elif cuda.is_cuda_array(A):
        return cp.asarray(A)
    else:
        raise Exception('as_cupy not implemented for objects of type "'+str(type(A))+'"')

def cp_trsv(L, b, x, transpose=False):
    '''
    If L is a lower triangular csr_matrix with added cu_ arrays
    (as used in the numba cuda_kernels), wraps the input in cupy and
    solves the lower triangular system Lx = b (if transpose==False),
    or the upper triangular system   L^Tx = b (if transpose==True).
    '''
    L_cp = as_cupy(L)
    x_cp = as_cupy(x)
    b_cp = as_cupy(b)
    if transpose:
        x_cp[:] = csp.linalg.spsolve_triangular(L_cp.T, b_cp, lower=False)
    else:
        x_cp[:] = csp.linalg.spsolve_triangular(L_cp, b_cp, lower=True)

