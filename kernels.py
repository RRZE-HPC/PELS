#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import sys
import os
from time import perf_counter

import numpy as np
import scipy
import numba

import sellcs

from numba import cuda
from cuda_kernels import *
from cupy_kernels import *

def available_gpus():
    if cuda is None or (os.environ.get('USE_CPU')=="1" or os.environ.get('USE_CPU')=="True"):
        return 0
    if cuda.is_available()==False:
        return 0
    return len(cuda.gpus)

def compile_all():
    n=10
    x=np.ones(n,dtype='float64')
    y=np.ones(n,dtype='float64')
    a=numba.float64(1.0)
    b=numba.float64(1.0)
    A1=(scipy.sparse.rand(n,n,0.6) + scipy.sparse.eye(n,n)).tocsr()
    L =scipy.sparse.tril(A1).tocsr()
    A2=sellcs.sellcs_matrix(A1, C=1, sigma=1)

    # compile GPU kernels:
    if available_gpus()==0:
        raise 'no GPU available!'
    x = to_device(x)
    tmp = from_device(x)
    y = to_device(x)
    A1 = to_device(A1)
    L = to_device(L)
    tmp= from_device(A1)
    A2 = to_device(A2)
    tmp = from_device(A2)
    init(x,a)
    z = clone(x)
    s=dot(x,y)
    axpby(a,x,b,y)
    spmv(A1,x,y)
    spmv(A2,x,y)
    trsv(L,x,y)
    diag_spmv(A1,x,y)
    reset_counters()

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

def to_host(A):
    if cuda.is_cuda_array(A):
        return A.copy_to_host()
    elif type(A)==scipy.sparse.csr_matrix or type(A)==sellcs.sellcs_matrix:
        if available_gpus()>0:
            A.indptr = A.cu_indptr.copy_to_host()
            A.data = A.cu_data.copy_to_host()
            A.indices = A.cu_indices.copy_to_host()
    return A


# total number of calls
calls = {'trsv': 0, 'spmv': 0, 'axpby': 0, 'dot': 0, 'init': 0}
# total elapsed time in seconds
time = {'trsv': 0, 'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total loaded data in GB
load = {'trsv': 0.0, 'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total stored data in GB
store = {'trsv': 0.0, 'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total floating point operations [GFlop]
flop = {'trsv': 0.0, 'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}

def reset_counters():
    for k in calls.keys():
        calls[k] = 0.0
        time[k] = 0.0
        load[k] = 0.0
        store[k] = 0.0
        flop[k] = 0.0

def same_array(x,y):
    '''
    returns 1 if the C pointer of the two arrays is identical, 0 otherwise
    '''
    if hasattr(x,'__cuda_array_interface__'):
        return int(x.__cuda_array_interface__['data'][0]==y.__cuda_array_interface__['data'][0])
    elif hasattr(x,'__array_interface__'):
        return int(x.__array_interface__['data'][0]==y.__array_interface__['data'][0])
    else:
        return False

def csr_spmv(valA,rptrA,colA, x, y):
        nrows = len(x)
        cu_csr_spmv.forall(nrows)(valA,rptrA,colA,x,y)
        cuda.synchronize()

def sell_spmv(valA,cptrA,colA, C, x, y):
        nchunks = len(cptrA)-1
        cu_sell_spmv[nchunks, C](valA, cptrA, colA, C, x, y)
        cuda.synchronize()


def spmv(A, x, y):
    t0 = perf_counter()

    if type(A)==scipy.sparse.csr_matrix:
        csr_spmv(A.cu_data, A.cu_indptr, A.cu_indices, x, y)
    elif type(A)==sellcs.sellcs_matrix:
        sell_spmv(A.cu_data, A.cu_indptr, A.cu_indices, A.C, x, y)
    else:
        raise TypeError('spmv wrapper only implemented for scipy.sparse.csr_matrix or sellcs.sellcs_matrix')
    t1 = perf_counter()
    time['spmv']  += t1-t0
    calls['spmv'] += 1
    load['spmv']  += 12*A.nnz+8*(A.shape[0]+A.shape[1])
    store['spmv'] += 8*A.shape[0]
    flop['spmv'] += 2*A.nnz

def trsv(L, b, x, transpose=False):
    '''
    Solves (i) Lx=b for x if transpose==False, or
          *ii) L^Tx=b     if transpose==True,

    where L is a sparse lower triangular matrix with
    non-zero diagonal. Elements in each row must be stored
    such that the diagonal is the last entry, but strict sorting
    is not required.
    '''

    t0 = perf_counter()

    if type(L)==scipy.sparse.csr_matrix:
        cp_trsv(L, b, x, transpose)
    elif type(L)==sellcs.sellcs_matrix:
        raise Exception('trsv only implemented for csr matrices so far')
    else:
        raise TypeError('trsv wrapper only implemented for scipy.sparse.csr_matrix or sellcs.sellcs_matrix')
    t1 = perf_counter()
    time['trsv']  += t1-t0
    calls['trsv'] += 1
    load['trsv']  += 12*L.nnz+8*(L.shape[0]+L.shape[1])
    store['trsv'] += 8*L.shape[0]
    flop['trsv'] += 2*L.nnz

def diag_spmv(A, x, y):
    cu_vscale.forall(x.size)(A.cu_data, x, y)


def csr_trsv(valL,rptrL,colL, x, b, transpose=False):
        '''
        Call CUDA CSR triangular solve kernel. See trsv
        function for description (and use it as entry point, not this one)
        '''
        nrows = len(x)
        flag = cuda.device_array(x.shape, dtype=np.int8)
        init(flag, np.int8(0))
        if transpose==False:
            cu_csr_trsv.forall(nrows)(valL,rptrL, colL, x,b, flag)
        else:
            cu_csr_trsv_trans.forall(nrows)(valL,rptrL, colL, x, b, flag)
        cuda.synchronize()

def clone(v):
    w = None
    w = cuda.device_array(shape=v.shape,dtype=v.dtype)
    return w

def permute_csr(X, perm):
    if type(X) == scipy.sparse.csr_matrix:
        data, indices, indptr = cpu.permute_csr_arrays(perm, X.data, X.indptr, X.indices)
        A = scipy.sparse.csr_matrix((data, indices, indptr), shape=X.shape)
        return A
    else:
        print("Error: permute_csr only applicable for scipy csr matrices. Retrning unpermuted matrix")
        return X

def copy(X):
    '''
    Copy a vector or matrix (csr_matrix or sellcs_matrix)
    that may live on a GPU, and assure first-touch initialization
    on the CPU.
    '''
    Y = cuda.device_array_like(X)
    Y[:] = X[:]
    return Y

def init(v, val):
    t0 = perf_counter()
    cu_init.forall(v.size)(v,val)
    cuda.synchronize()
    t1 = perf_counter()
    calls['init'] += 1
    time['init']  += t1-t0
    store['init'] += 8*v.size

def axpby(a,x,b,y):
    t0 = perf_counter()
    cu_axpby.forall(x.size)(a,x,b,y)
    cuda.synchronize()
    t1 = perf_counter()
    time['axpby']  += t1-t0
    calls['axpby'] += 1
    load['axpby']  += (2-same_array(x,y))*8*x.size
    store['axpby'] += 8*x.size
    flop['axpby'] += 2*x.size

def dot(x,y):
    t0 = perf_counter()
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
    s[0] = 0.0
    cu_dot[BlocksPerGrid,ThreadsPerBlock](x,y,s)
    cuda.synchronize()
    t1 = perf_counter()
    time['dot']  += t1-t0
    calls['dot'] += 1
    load['dot']  += (2-same_array(x,y))*8*x.size
    flop['dot'] += 2*x.size
    return s.copy_to_host()[0]

def perf_report():
    '''
    After running a solver, print a performance summary of the
    kernels in this module (dot, axpby, spmv...). The argument 'type'
    should be either 'cpu' or 'gpu', dependning on which hardware you
    ran. It can be used to get some benchmark values from files cpu.json or
    gpu.json, but we currently ignore them.
    '''

    device = cuda.get_current_device()

    # The 'name' attribute usually contains the model string (e.g., 'NVIDIA GeForce RTX 3080')
    print('Hardware: '+str(device.name))

    # total measured time
    t_tot  = 0
    # model prediction
    t_mod  = 0
    # total number of functions called
    total_calls = 0

    print('--------\t-----\t---------------\t---------------\t---------------')
    print('kernel  \tcalls\t bw_estimate   \t t_meas        \t t_meas/call   ')
    print('========\t=====\t===============\t===============\t===============')
    for kern in ('dot', 'axpby', 'spmv', 'trsv'):
        if calls[kern]>0:
            t_tot += time[kern]
            total_calls += calls[kern]
            print('%8s\t%5d\t%8.4g GB/s\t%8.4g s \t%8.4g s \t'%
                    (kern, calls[kern], (load[kern]+store[kern])*1e-9/time[kern], time[kern], time[kern]/calls[kern]))

    print('--------\t-----\t---------------\t---------------')
    print('%8s\t     \t               \t               \t %8.4g s '%('Total',t_tot))
    print('--------\t-----\t---------------\t---------------')
