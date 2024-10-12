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
from numba import jit, prange, get_num_threads, float64
import scipy
import json

def memory_benchmarks():
    benchmarks = {'label': 'undefined', 'triad': 0, 'load': 0, 'store': 0, 'copy': 0}
    try:
        with open('cpu.json', 'r') as f:
            benchmarks = json.load(f)
    except:
        return benchmarks
    nthreads = get_num_threads()
    ncores_data = benchmarks['cores']
    nnuma    = (nthreads+ncores_data-1)//ncores_data
    if nthreads == ncores_data:
        return benchmarks
    else:
        result = benchmarks.copy()
        for k in result.keys():
            if k != 'label':
                result[k] *=nnuma
        return result


@jit(nopython=True, parallel=True)
def copy_vector(x):
    y = np.empty_like(x)
    for i in prange(x.size):
        y[i] = x[i]
    return y

@jit((float64[:],float64[:],float64[:]),nopython=True, parallel=True)
def vscale(v, x, y):
    '''
    Vector scaling y[i] = v[i]*x[i]
    '''
    for i in prange(x.size):
        y[i] = v[i]*x[i]

@jit(nopython=True, parallel=True)
def copy_csr_arrays(Adata, Aindptr, Aindices):
    data = np.empty_like(Adata)
    indices = np.empty_like(Aindices)
    indptr = np.empty_like(Aindptr)
    nrows = len(indptr)-1
    nnz   = len(Adata)
    for i in prange(nrows):
        indptr[i] = Aindptr[i]
        indptr[i+1] = Aindptr[i+1]
        for j in range(indptr[i], indptr[i+1]):
            data[j] = Adata[j]
            indices[j] = Aindices[j]
    return data, indices, indptr

@jit(nopython=True, parallel=True)
def csr_spmv(valA,rptrA,colA, x, y):
    '''
    Usage:
      - if A is a scipy.sparse.csr_matrix,
      you can get the components by valA = A.data; rptrA = cptrA; colA = A.indices.
      - x and y numpy arrays of size A.shape[0] and A.shape[1], respectively.

      Then this function returns y = A*x
    '''
    for row in prange(len(rptrA)-1):
        y[row] = 0
        for j in range(rptrA[row], rptrA[row+1]):
            y[row] += valA[j] * x[colA[j]]

@jit(nopython=True, parallel=True)
def sell_spmv(valA, cptrA, colA, C, x, y):
    '''
    Usage: sell_spmv(valA, cptrA, colA, C, x, y) computes y=A*x for a sellcs.sellcs_matrix A,
           where the members are extracted like this:
           valA, cptrA, colA, C = A.data, cptrA, A.indices, A.C
           Sorting of in and/or output vectors is **not performed** by this function:
           If A.sigma>1, the user is responsible to provided x in permuted form and
           "unpermute" y if desired.
    '''
    nchunks = len(cptrA)-1
    nrows = x.size
    for chunk in prange(nchunks):
        offs = cptrA[chunk]
        row0 = chunk*C
        row1 = min(row0+C, nrows)
        c    = row1-row0
        w    = (cptrA[chunk+1]-offs)//c
        #print('rows %d:%d, c=%d, w=%d'%(row0,row1,c,w))
        y[row0:row1] = 0
        for j in range(w):
                y[row0:row1] += valA[offs+j*c:offs+(j+1)*c] * x[colA[offs+j*c:offs+(j+1)*c]]


@jit(nopython=True, parallel=True)
def init(x, val):
    for i in prange(x.size):
        x[i]=val

@jit(nopython=True, parallel=True)
def axpby(a,x,b,y):
    for i in prange(x.size):
        y[i]=a*x[i]+b*y[i]

@jit(nopython=True, parallel=True)
def dot(x,y):
    s=0.0
    for i in prange(x.size):
        s += x[i]*y[i]
    return s

