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
from numba import cuda, float64, int32
import scipy
from math import *
import sellcs

################
# CUDA kernels #
################

@cuda.jit((float64, float64[:], float64, float64[:]))
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


@cuda.jit((float64[:], float64[:]))
def cu_vscale_inplace(v, x):
    '''
    Vector scaling x[i] = v[i]*x[i]
    '''
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    i = bx * bdx + tx
    if i < x.size:
        x[i] = v[i]*x[i]

@cuda.jit((float64[:], float64[:], float64[:]))
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

# Our dot product uses three stages:
# 1) A "grid-stride oop" to reduce the number of summands so they fit
#    into shared emmory while reading x and y with high memory bandwidth,
# 2) A binary tree reduction in shared memory to retain one number per thread block,
# 3) An atomic add operation per thread block to produce the final sum.
@cuda.jit((float64[:], float64[:], float64[:]))
def cu_dot(x,y,s):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x

    # we perform the operation for all threads and then move to the next
    # chunk, where the chunk size (stride) is given by the total number
    # of threads:
    stride = cuda.gridDim.x*bdx

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

@cuda.jit((float64[:], int32[:], int32[:], float64[:], float64[:]))
def cu_csr_spmv(valA,rptrA,colA, x, y):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    i = bx * bdx + tx
    if i < x.size:
        y[i] = 0.0
        for j in range(rptrA[i], rptrA[i+1]):
            y[i] += valA[j]*x[colA[j]]

@cuda.jit((float64[:], int32[:], int32[:], int32, float64[:], float64[:]))
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

