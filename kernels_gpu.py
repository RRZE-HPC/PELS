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
import numba
from numba import cuda, float64
import scipy
from math import *
import sellcs
import json

def memory_benchmarks():
    benchmarks = {'label': 'undefined', 'triad': 0, 'load': 0, 'store': 0, 'copy': 0}
    try:
        with open('gpu.json', 'r') as f:
            benchmarks = json.load(f)
    except:
        pass
    return benchmarks

def to_device(A):
    if type(A) == scipy.sparse.csr_matrix or type(A) == sellcs.sellcs_matrix:
        A.cu_data = cuda.to_device(A.data)
        A.cu_indptr = cuda.to_device(A.indptr)
        A.cu_indices = cuda.to_device(A.indices)
        return A
    elif type(A) == scipy.sparse.dia_matrix:
        A.cu_data = cuda.to_device(A.data.reshape(A.data.size*A.offsets.size))
        A.cu_offsets = cuda.to_device(A.offsets)
        return A
    else:
        return cuda.to_device(A)

def from_device(A):
    if type(A) == scipy.sparse.csr_matrix or type(A) == sellcs.sellcs_matrix:
        A.data = A.cu_data.copy_to_host()
        A.indices = A.cu_indices.copy_to_host()
        A.indptr = A.cu_indptr.copy_to_host()
        return A
    else:
        return cuda.to_device(A)

################
# CUDA kernels #
################

@cuda.jit
def cu_axpby(a,x,b,y):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    i = bx * bdx + tx
    if i < x.size:
        y[i]=a*x[i]+b*y[i]

@cuda.jit
def cu_init(x, val):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    i = bx * bdx + tx
    if i < x.size:
        x[i] = val


@cuda.jit((float64[:],float64[:],float64[:]))
def cu_vscale(v, x, y):
    '''
    Vector scaling y[i] = v[i]*x[i]
    '''
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    i = bx * bdx + tx
    if i < x.size:
        y[i] = v[i]*x[i]

# dot product is complicated on GPU's...
# we first define a reduction operation that will be
# used to get the final result over all thread blocks,
# the actual kernel recursively sums up components into
# a memory location shared within a block.

@cuda.jit((float64[:],float64[:],float64[:]))
def cu_dot(x,y,s):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x

    # we perform the operation for all threads and then move to the next
    # chunk, where the chunk size (stride) is given by the total number
    # of threads:
    stride = cuda.gridDim.x*bdx

    s[0] = 0.0
    s_shared = cuda.shared.array(shape=(128), dtype=float64) # array with one element per thread in the block
    s_shared[tx] = 0.0
    s_local = 0.0 # scalar value per thread

    i = bx * bdx + tx
    while i < x.size:
        s_local += x[i]*y[i]
        i += stride
    s_shared[tx] = s_local
    cuda.syncthreads()

    i = tx
    stride = bdx//2
    while stride != 0:
        if i < stride:
            s_shared[i] += s_shared[i+stride]
        stride = stride//2
        cuda.syncthreads()

    if tx == 0:
        # atomic operation to avoid race condition,
        # thread 0 from each block sums it's local contribution
        # into s[0]:
        cuda.atomic.add(s, 0, s_shared[0])

@cuda.jit
def cu_csr_spmv(valA,rptrA,colA, x, y):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    i = bx * bdx + tx
    if i < x.size:
        y[i] = 0.0
        for j in range(rptrA[i], rptrA[i+1]):
            y[i] += valA[j]*x[colA[j]]

@cuda.jit
def cu_sell_spmv(valA, cptrA, colA, C, x, y):
    '''
    This kernel assumes that it is launched with the block size equal to the
    chunk-size C of the SELL-C-sigma matrix represented by [cptrA,valA,colA]
    '''
    tx = cuda.threadIdx.x
    chunk = cuda.blockIdx.x
    assert(C == cuda.blockDim.x)

    row   = chunk*C + tx
    offs  = cptrA[chunk]

    nchunks = len(cptrA)-1
    nrows = x.size

#    y_shared = cuda.shared.array(shape=(cuda.blockDim.x), dtype=float64) # array with one element per thread in the block
    y_shared = cuda.shared.array(shape=(128), dtype=float64) # array with one element per thread in the block

    if row>=nrows:
        return
    c = min(C,nrows-chunk*C)
    w    = (cptrA[chunk+1]-offs)//c
    y_shared[tx] = 0
    for j in range(w):
        y_shared[tx] += valA[offs+j*c+tx] * x[colA[offs+j*c+tx]]
    y[row] = y_shared[tx]

################################################################################
# Wrapper functions to expose to the user.                                     #
# Note that these will work both with regular (host-side) numpy arrays         #
# and numba device arrays, but that the latter avoids the cost of trans-       #
# ferring data to/from the GPU.                                                #
# In order to allow timing the operations, we add a sync statement after       #
# the kernel runs, but this is not strictly necessary and we could add a       #
# 'nosync' parameter to each function to allow optimization of complete        #
# algorithms.                                                                  #
################################################################################

def axpby(a,x,b,y):
    cu_axpby.forall(x.size)(a,x,b,y)
    cuda.synchronize()

def init(v, val):
    cu_init.forall(v.size)(v,val)
    cuda.synchronize()

def vscale(v, x, y):
    cu_vscale.forall(x.size)(v, x, y)
    '''
    Vector scaling x[i] = v[i]*x[i]
    '''

def dot(x,y):
    ##return cu_dot2.forall(x.size)(x,y)
    #
    # note: we could do "forall" here as well,
    # but the cu_dot implementation requires that
    # the threads per block are a power of two. This
    # is almost certainly the case by default, but to be
    # sure, we enforce it here.
    ThreadsPerBlock = 128
    BlocksPerGrid   =1024
#    BlocksPerGrid   = min(32, (x.size+ThreadsPerBlock-1)//ThreadsPerBlock)
    s = cuda.device_array(shape=(1), dtype=np.float64)
    cu_dot[BlocksPerGrid,ThreadsPerBlock](x,y,s)
    return s.copy_to_host()[0]

def csr_spmv(valA,rptrA,colA, x, y):
        nrows = len(x)
        cu_csr_spmv.forall(nrows)(valA,rptrA,colA,x,y)
        cuda.synchronize()
        #print(y.copy_to_host())


def sell_spmv(valA,cptrA,colA, C, x, y):
        nchunks = len(cptrA)-1
        cu_sell_spmv[nchunks, C](valA, cptrA, colA, C, x, y)
        cuda.synchronize()


if __name__ == '__main__':

    from timeit import timeit

    print('Numba GPU example. Available device(s):')
    print(cuda.gpus)

    N=2**26
    ntimes=30

    x_host = np.random.rand(N)
    y_host = np.random.rand(N)
    z_host = np.empty_like(x_host)
    a = 42.9

    # allocate device memory and copy x, y and z to the GPU
    x = cuda.to_device(x_host)
    y = cuda.to_device(y_host)
    z = cuda.to_device(z_host)

    # compile once and test results on host
    axpy(a,x,y,z)
    print('Error axpy: %e'%(np.linalg.norm(z.copy_to_host() - axpy_host(a, x_host, y_host, z_host), 2)))

    s = dot(x,y)
    print('Error   dot: %e'%(abs(s-np.dot(x_host,y_host))))

    # measure

    t = timeit(stmt='zz=axpy(a,x,y,z)', globals=globals(), number=ntimes)
    print ('axpy: %8.4f GB/s\n'%(4*N*ntimes*8.0e-9/t))

    t = timeit(stmt='s=dot(x,y)', globals=globals(), number=ntimes)
    print ('dot: %8.4f GB/s\n'%(2*N*ntimes*8.0e-9/t))

    # get the "result"
    z.copy_to_host(z_host)
