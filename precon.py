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
from scipy.sparse.linalg import spilu
from time import perf_counter

import cupy as cp
from cupy.cuda import cusolver, cusparse
from cupyx.scipy.sparse.linalg import spilu as cupy_spilu

import kernels
from cuda_precon import *
from cupy_kernels import as_cupy

from numba import cuda

# total number of calls
calls = {'setup': 0, 'apply': 0}
# total elapsed time in seconds
time = {'setup': 0, 'apply': 0.0, 'axpby': 0.0}

def invert(v):
    cu_invert.forall(v.size)(v)
    cuda.synchronize()

def neumann(A0, k, v0, x):
    r'''
    Apply degree-k Neumann polynomial in matrix A = (I-A0):

    x = A^{-1} v0 = (I-A0)^{-1} v0 \approx \sum_{j=0}^k (A0^j v0)

    '''
    v = kernels.clone(v0)
    A0v = kernels.clone(v0)
    # set x = v0 (= A0^0 v0)
    axpby(1.0,v0, 0.0,x)
    for _ in range(k):
        # A0v = A0^{j+1}v0
        spmv(A0, v, A0v)
        # x += v0 to yield x_k
        axpby(1.0, A0v, 1.0, x)
        # swap the vectors
        A0v, v = v, A0v

class FastTrsv:

    def __init__(self, L):
        '''
        Run analysis to optimize triangular solves (pass the output to cp_trsv)
        '''
        n = L.shape[0]
        nnz = L.nnz
        self.shape = L.shape
        self.nnz   = L.nnz
        self.L_cp = as_cupy(L)
        self.L_cp.sort_indices()

        # Create Sparse Matrix Descriptor for L
        self.handle = cusparse.create()
        self.matA = cusparse.createCsr(
                n, n, nnz,
                self.L_cp.indptr.data.ptr, self.L_cp.indices.data.ptr, self.L_cp.data.data.ptr,
                cusparse.CUSPARSE_INDEX_32I, cusparse.CUSPARSE_INDEX_32I,
                cusparse.CUSPARSE_INDEX_BASE_ZERO, cp.cuda.runtime.CUDA_R_64F
                )

        # Specify that A is Lower-Triangular and Non-Unit diagonal
        fill_mode = np.array(cusparse.CUSPARSE_FILL_MODE_LOWER, dtype=np.int32)
        diag_type = np.array(cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT, dtype=np.int32)
        cusparse.spMatSetAttribute(self.matA, cusparse.CUSPARSE_SPMAT_FILL_MODE, fill_mode)
        cusparse.spMatSetAttribute(self.matA, cusparse.CUSPARSE_SPMAT_DIAG_TYPE, diag_type)

        # Create Dense Matrix Descriptors for B and X
        # Note: SpSM expects row-major or column-major layouts specified explicitly
        # We keep these descriptors alive as required by the Generic API.
        x_cp = cp.zeros((n,), dtype=cp.float64)
        b_cp = cp.zeros((n,), dtype=cp.float64)
        self.matX = cusparse.createDnMat(n, 1, n, x_cp.data.ptr, cp.cuda.runtime.CUDA_R_64F,
                                         cusparse.CUSPARSE_ORDER_COL)
        self.matB = cusparse.createDnMat(n, 1, n, b_cp.data.ptr, cp.cuda.runtime.CUDA_R_64F,
                                         cusparse.CUSPARSE_ORDER_COL)

        # Create an opaque SpSM descriptor to hold the cached analysis phase data
        self.spsmDescr  = cusparse.spSM_createDescr()
        # and another for the transposed operation, x = L^{-T}b
        self.spsmDescrT = cusparse.spSM_createDescr()

        self.alpha = np.array(1.0, dtype='float64')

        ##########################################################
        # Analysis Phase -- run for standard and transposed case #
        ##########################################################

        # Query buffer size needed for the analysis and execution steps
        bufferSize = cusparse.spSM_bufferSize(
                self.handle, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
                computeType=cp.cuda.runtime.CUDA_R_64F, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
                spsmDescr=self.spsmDescr)
        bufferSizeT = cusparse.spSM_bufferSize(
                self.handle, cusparse.CUSPARSE_OPERATION_TRANSPOSE,
                cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
                computeType=cp.cuda.runtime.CUDA_R_64F, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
                spsmDescr=self.spsmDescrT)

        # The externalBuffer must be preserved and remain unmodified between the analysis and solve
        # phases. Since we have two separate analyses (non-transpose and transpose), we need two
        # buffers.
        self.buffer = cp.empty(bufferSize, dtype=cp.int8)
        self.bufferT = cp.empty(bufferSizeT, dtype=cp.int8)

        cusparse.spSM_analysis(
            self.handle, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
            computeType=cp.cuda.runtime.CUDA_R_64F, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr=self.spsmDescr, externalBuffer=self.buffer.data.ptr)

        cusparse.spSM_analysis(
            self.handle, cusparse.CUSPARSE_OPERATION_TRANSPOSE,
            cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
            computeType=cp.cuda.runtime.CUDA_R_64F, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr=self.spsmDescrT, externalBuffer=self.bufferT.data.ptr)

    def __del__(self):
        cusparse.spSM_destroyDescr(self.spsmDescr)
        cusparse.spSM_destroyDescr(self.spsmDescrT)
        cusparse.destroyDnMat(self.matB)
        cusparse.destroyDnMat(self.matX)
        cusparse.destroySpMat(self.matA)
        cusparse.destroy(self.handle)

    def apply(self, b, x, transpose=False):
        '''
        Solves L x   = b (if transpose==False), or
               L^T x = b (if transpose==True) for x,

               where L is the lower triangular matrix passed to the constructor.
        '''
        t0 = perf_counter()
        x_cp = as_cupy(x)
        b_cp = as_cupy(b)

        # Update the descriptors to point to the current data
        cusparse.dnMatSetValues(self.matX, x_cp.data.ptr)
        cusparse.dnMatSetValues(self.matB, b_cp.data.ptr)

        if transpose:
            cp_transpose = cusparse.CUSPARSE_OPERATION_TRANSPOSE
            spsmDescr = self.spsmDescrT
            buffer_ptr = self.bufferT.data.ptr
        else:
            cp_transpose = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
            spsmDescr = self.spsmDescr
            buffer_ptr = self.buffer.data.ptr

        cusparse.spSM_solve(
                self.handle, cp_transpose, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
                computeType=cp.cuda.runtime.CUDA_R_64F, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
                spsmDescr=spsmDescr, externalBuffer=buffer_ptr)

        cuda.synchronize()
        t1 = perf_counter()
        kernels.time['trsv']  += t1-t0
        kernels.calls['trsv'] += 1
        kernels.load['trsv']  += 12*self.nnz+8*(self.shape[0]+self.shape[1])
        kernels.store['trsv'] += 8*self.shape[0]
        kernels.flop['trsv'] += 2*self.nnz

###################
# Preconditioners #
###################

class Jacobi:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        t0 = perf_counter()
        self.D_inv = kernels.to_device(A.diagonal())
        invert(self.D_inv)
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Diagonal scaling, v = D^{-1}w
        '''
        t0 = perf_counter()
        kernels.vscale(self.D_inv, w, v)
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

class SymmetricGaussSeidel:
    '''
    Use a classical symmetric Gauss-Seidel step as a preconditioner:

    A = L+D+L^T
    M = (L+D) D (L+D)^T, resp.
    M^{-1} = (L+D)%{-T} D^{-1} (L+D)^{-1}.

    If fast_trsv=True, use cusparse to analyze the pattern of L+D
    at construction time, giving faster solves in "apply"
    '''
    def __init__(self, A, fast_trsv=False):
        t0 = perf_counter()
        self.D = kernels.to_device(A.diagonal())
        self.LplusD = kernels.to_device(scipy.sparse.tril(A).tocsr())
        if fast_trsv:
            self.fast_trsv = FastTrsv(self.LplusD)
        self.v_tmp = kernels.to_device(np.zeros(A.shape[0]))
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Symmetric Gauss-Seidel: v = (L+D)^{-T} D (L+D)^{-1} w
        '''
        t0 = perf_counter()
        if hasattr(self, 'fast_trsv'):
            self.fast_trsv.apply(w, self.v_tmp)
        else:
            kernels.trsv(self.LplusD, w, self.v_tmp)
        kernels.vscale_inplace(self.D, self.v_tmp)
        if hasattr(self, 'fast_trsv'):
            self.fast_trsv.apply(self.v_tmp, v, transpose=True)
        else:
            kernels.trsv(self.LplusD, self.v_tmp, v, transpose=True)
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

class IChol:
    '''
    Incomplete Cholesky factorization preconditioner.

    Given a symmetric and positive definite (spd) matrix A,
    computes A \approx LL^T, where L is lower triangular. By default, L
    inherits the sparsity pattern of the lower triangular part of A (zero-fill, or ILU0).

    The preconditioner is applied as a sequence of a forward triangular solve with L and
    a backward triangular solve with L^T.

    **Improving the preconditioner**

    Two parameters affect the approximation quality and fill-in allowed in L:

    - droptol: remove newly generated elements in during elimination if they are
      smaller than droptol (in absolute value, relative to the diagonal)
      (default 0, no drop-by-value)
    - fill: restrict newly created entries ("fill-in") to positions the sparsity pattern of A^{fill}
      (default 1, 'zero-fill ILU')

    If fast_trsv=True, uses cusparse to analyze the pattern of L, resulting in faster triangular solves during "apply".

    **Avoiding triangular solves**

    If poly_k>=0 is given, triangular solves are replaced by a degree <poly_k> Neumann polynomial,
    resulting in 2x <poly_k> spmvs with triangular matrices instead of two triangular solves.
    This reduces the approximation quality of the preconditioner, but eliminates data dependencies and thus imporoves
    performance on the GPU.

    Implementation note: The factorization is done on the host (CPU) using scipy, and in fact
    uses an ILU algorithm without pivoting, so the 'setup' phase is highly non-optimal.
    The triangular solves rely on the CuPy implementation (via trsv in kernels.py).

    Example:

    A  = matrix_generator.create_matrix('Laplace500x500')
    IC = IChol(A, fill=3, droptol=0.01) # uses forward/backward solves
    y  = kernels.to_device(numpy.random.random(A.shape[0]))
    x  = kernels.to_device(numpy.zeros(A.shape[0]))
    IC.apply(y, x)
    '''

    def __init__(self, A, fill=1, droptol=0.0, fast_trsv=False, poly_k=-1):
        '''
        Factor A \approx LL^T on the host.

        Input:

          A: scipy.sparse.csr_matrix, only host-sde is accessed.

        Output:

          self.L: lower triangular factor, copied to device using "kernels.to_device"
        '''

        t0 = perf_counter()
        self.shape = A.shape
        self.dtype = A.dtype
        # For poly_k>0 we use a Neumann polynomial to approximate the triangular solves
        self.poly_k = poly_k
        # create a temporary vector for the 'apply' function:
        self.v_tmp = kernels.to_device(np.zeros(A.shape[0]))

        # Compute factorization using SciPy's ILU0. Because A is spd,
        # we (i) do not need pivoting, and (b) can scale the result such that
        # L=U^T
        ilu = spilu(A.T, drop_tol=droptol, fill_factor=fill, permc_spec='NATURAL', diag_pivot_thresh=0.0)

        L = ilu.L
        d = ilu.U.diagonal()

        if poly_k<0:
            # scale such that A \approx LL^T
            D = scipy.sparse.diags(np.sqrt(d))
            L = (L@D).tocsr()
            self.L = kernels.to_device(L)
            if fast_trsv:
                self.fast_trsv = FastTrsv(self.L)
        else:
            d_inv = 1.0/d
            self.d_inv = kernels.to_device(d_inv)

            # store the (negative) factor L and its explicit transpose, but skip the diagonal:
            # L = (I - L0), L^T = (I - L0t)
            # for implementing the Neumann polynomial approximation if the inverse (see 'apply')
            L0  = scipy.sparse.tril(-L,k=-1, format='csr')
            L0t = L0.T.tocsr()
            self.L0 = kernels.to_device(L0)
            self.L0t = kernels.to_device(L0t)
            self.w_tmp = kernels.clone(self.v_tmp)


        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Apply the preconditioner M^{-1} to a vector, i.e.,
        Mv = w -> LL^Tv = w
               -> v = L^{-T} L^{-1} w
        '''
        t0 = perf_counter()
        if not (cuda.is_cuda_array(v) and
                cuda.is_cuda_array(w)):
            raise Exception('IChol preconditioner requires vectors to be cuda arrays')

        if self.poly_k<0:
            if hasattr(self, 'fast_trsv'):
                self.fast_trsv.apply(w, self.v_tmp, False)
            else:
                kernels.trsv(self.L, w, self.v_tmp, False)
            if hasattr(self, 'fast_trsv'):
                self.fast_trsv.apply(self.v_tmp, v, True)
            else:
                kernels.trsv(self.L, self.v_tmp, v, True)
        else:
            # Use the degree-k Neumann polynomial to approximate the two triangular solves.
            #
            # With L = (I-L0), L^T = (I-L0t), A \approx LDL^T
            # Solve Av = (LDL^T)v = w as

            # 1. v_tmp = L^{-1}w = (I-L0)^{-1}w \approx \sum_j L0^j w
            neumann(self.L0, self.poly_k, w, self.v_tmp)

            # 2. W_tmp = D^{-1}v_tmp
            kernels.vscale(self.d_inv, self.v_tmp, self.w_tmp)

            # 3. v = L^{-T}w_tmp \approx \sum_{j=0}^k L0t^j w_tmp
            neumann(self.L0t, self.poly_k, self.w_tmp, v)

        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

