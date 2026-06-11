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
import pymetis

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

def cusparse_neumann(matA, k, v0, x, n, dtype, nnz, handle=None):
    '''
    Apply degree-k Neumann polynomial in matrix M = (I-A) using cuSPARSE SpMV:
    x = M^{-1} v0 = (I-A)^{-1} v0 \approx \sum_{j=0}^k (A^j v0)

    This uses the iteration x_{j+1} = v0 + A * x_j, with x_0 = v0.
    Implemented using cusparseSpMV with alpha=1, beta=1.
    
    Input:
      matA: cusparseSpMatDescr_t for matrix A (should have unit diagonal attributes if needed)
      k: degree of polynomial
      v0: initial RHS vector
      x: output vector
      n: dimension of the matrix
      dtype: numpy dtype (float32/float64)
      nnz: number of non-zeros in A
      handle: cusparse handle
    '''
    if k <= 0:
        axpby(1.0, v0, 0.0, x)
        return

    t0 = perf_counter()

    if handle is None:
        handle = cusparse.create()
        own_handle = True
    else:
        own_handle = False

    cuda_dtype = cp.cuda.runtime.CUDA_R_64F if dtype == np.float64 else cp.cuda.runtime.CUDA_R_32F

    v_tmp = kernels.clone(v0)
    # x_0 = v0
    axpby(1.0, v0, 0.0, x)

    alpha = np.array(1.0, dtype=dtype)
    beta = np.array(1.0, dtype=dtype)

    # Vector descriptors
    def get_ptr(v):
        if hasattr(v, 'data'): return v.data.ptr
        return v.__cuda_array_interface__['data'][0]

    matX = cusparse.createDnVec(n, get_ptr(x), cuda_dtype)
    matV = cusparse.createDnVec(n, get_ptr(v_tmp), cuda_dtype)

    bufSize = cusparse.spMV_bufferSize(handle, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       alpha.ctypes.data, matA, matX, beta.ctypes.data, matV,
                                       cuda_dtype, cusparse.CUSPARSE_SPMV_ALG_DEFAULT)
    buf = cp.empty(bufSize, dtype=cp.int8)

    for _ in range(k):
        # v_tmp = v0
        axpby(1.0, v0, 0.0, v_tmp)
        # v_tmp = 1.0 * A * x + 1.0 * v_tmp  =>  v_tmp = v0 + A * x
        cusparse.dnVecSetValues(matX, get_ptr(x))
        cusparse.dnVecSetValues(matV, get_ptr(v_tmp))
        cusparse.spMV(handle, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                      alpha.ctypes.data, matA, matX, beta.ctypes.data, matV,
                      cuda_dtype, cusparse.CUSPARSE_SPMV_ALG_DEFAULT, buf.data.ptr)
        # x = v_tmp
        axpby(1.0, v_tmp, 0.0, x)

    cusparse.destroyDnVec(matX)
    cusparse.destroyDnVec(matV)
    if own_handle:
        cusparse.destroy(handle)

    t1 = perf_counter()
    if 'spmv' in kernels.time:
        kernels.time['spmv']  += t1-t0
        kernels.calls['spmv'] += k

def apply_metis_preordering(A, x_ex, b):
    '''
    Given a linear system A x_ex = b, where x_ex is the exact solution,
    returns A_p, x_ex_p, b_p s.t. A_p x_ex_p = b_p, and the '_p' matrices
    /vectors are consistent permutations of there input counterparts.
    A_p retains symmetry and should have a more favorable pattern for subsequent
    (incomplete) factorization, as with IChol.
    '''
    # Compute Nested Dissection permutation using PyMetis
    # Metis expects a graph WITHOUT self-loops (diagonal entries).
    n = A.shape[0]
    adjacency = [
             [j for j in A.indices[A.indptr[i] : A.indptr[i+1]] if j != i]
                for i in range(n)
            ]

    # iperm[i] is the vertex at position i of the new ordering.
    _, p = pymetis.nested_dissection(adjacency)
    p = np.array(p, dtype=int)

    # Pre-permute the matrix A_p = A[p, p]
    A_p    = A[p, :][:, p]
    x_ex_p = x_ex[p]
    b_p    = b[p]
    return A_p, x_ex_p, b_p






class FastTrsv:

    def __init__(self, L, cusparse_handle=None, descrL=None, poly_k=-1):
        '''
        Run analysis to optimize triangular solves (pass the output to cp_trsv)
        If you already have a cusparse handle and/or matrix descriptor (e.g. from the factorization phase),
        you can pass them in here to avoid duplicate work.
        '''
        n = L.shape[0]
        nnz = L.nnz
        self.shape = L.shape
        self.nnz   = L.nnz
        self.dtype = L.dtype
        self.poly_k = poly_k
        self.own_handle = False
        self.own_matA = False

        if cusparse_handle is None:
            self.handle = cusparse.create()
            self.own_handle = True
        else:
            self.handle = cusparse_handle

        # Determine CUDA data type from L.dtype
        if self.dtype == np.float32:
            self.cuda_dtype = cp.cuda.runtime.CUDA_R_32F
        elif self.dtype == np.float64:
            self.cuda_dtype = cp.cuda.runtime.CUDA_R_64F
        else:
            raise TypeError(f"Unsupported dtype {self.dtype}. Expected float32 or float64.")

        # Always use a Generic API SpMatDescr for spSM
        # To be safe, we always create our own SpMatDescr from L.
        self.L_cp = as_cupy(L)
        self.L_cp.sort_indices()

        # Determine index dtypes (32 vs 64 bit)
        if self.L_cp.indptr.dtype == np.int32:
            ptr_type = cusparse.CUSPARSE_INDEX_32I
        else:
            ptr_type = cusparse.CUSPARSE_INDEX_64I

        if self.L_cp.indices.dtype == np.int32:
            idx_type = cusparse.CUSPARSE_INDEX_32I
        else:
            idx_type = cusparse.CUSPARSE_INDEX_64I

        # Create Sparse Matrix Descriptor for L
        self.matA = cusparse.createCsr(
            n, n, nnz,
            self.L_cp.indptr.data.ptr, self.L_cp.indices.data.ptr, self.L_cp.data.data.ptr,
            ptr_type, idx_type,
            cusparse.CUSPARSE_INDEX_BASE_ZERO, self.cuda_dtype
            )
        self.own_matA = True

        # Specify that A is Lower-Triangular and Non-Unit diagonal
        fill_mode = np.array(cusparse.CUSPARSE_FILL_MODE_LOWER, dtype=np.int32)
        diag_type = np.array(cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT, dtype=np.int32)
        cusparse.spMatSetAttribute(self.matA, cusparse.CUSPARSE_SPMAT_FILL_MODE, fill_mode)
        cusparse.spMatSetAttribute(self.matA, cusparse.CUSPARSE_SPMAT_DIAG_TYPE, diag_type)

        # Create Dense Matrix Descriptors for B and X
        # We MUST keep the dummy vectors alive because the descriptors (and analysis)
        # may depend on them until they are updated in 'apply'.
        self._x_dummy = cp.zeros((n,), dtype=self.dtype)
        self._b_dummy = cp.zeros((n,), dtype=self.dtype)
        self.matX = cusparse.createDnMat(n, 1, n, self._x_dummy.data.ptr, self.cuda_dtype,
                                         cusparse.CUSPARSE_ORDER_COL)
        self.matB = cusparse.createDnMat(n, 1, n, self._b_dummy.data.ptr, self.cuda_dtype,
                                         cusparse.CUSPARSE_ORDER_COL)

        # Create an opaque SpSM descriptor to hold the cached analysis phase data
        self.spsmDescr  = cusparse.spSM_createDescr()
        # and another for the transposed operation, x = L^{-T}b
        self.spsmDescrT = cusparse.spSM_createDescr()

        self.alpha = np.array(1.0, dtype=self.dtype)

        # Components for Neumann polynomial: L = I - L0
        # If poly_k >= 0, we initialize the required matrix components.
        self._L0 = None
        self._L0t = None
        self._matL0 = None
        self._matL0t = None
        if self.poly_k >= 0:
            # Extract diagonal and normalize: L = D(I - L0)
            L_cpu = self.L_cp.get()
            d = L_cpu.diagonal()
            self._d_inv = kernels.to_device(1.0 / d)
            
            # L0 = (D - L) @ D^{-1}
            D = scipy.sparse.diags(d)
            L0_cpu = (D - L_cpu) @ scipy.sparse.diags(1.0/d)
            self._L0 = kernels.to_device(L0_cpu.tocsr())
            self._L0t = kernels.to_device(L0_cpu.T.tocsr())
            
            self._v_poly = kernels.to_device(np.zeros(n, dtype=self.dtype))

            # Create Generic API descriptors for L0 and L0t
            self._matL0 = cusparse.createCsr(
                n, n, self._L0.nnz,
                self._L0.cu_indptr.__cuda_array_interface__['data'][0],
                self._L0.cu_indices.__cuda_array_interface__['data'][0],
                self._L0.cu_data.__cuda_array_interface__['data'][0],
                ptr_type, idx_type, cusparse.CUSPARSE_INDEX_BASE_ZERO, self.cuda_dtype
                )
            self._matL0t = cusparse.createCsr(
                n, n, self._L0t.nnz,
                self._L0t.cu_indptr.__cuda_array_interface__['data'][0],
                self._L0t.cu_indices.__cuda_array_interface__['data'][0],
                self._L0t.cu_data.__cuda_array_interface__['data'][0],
                ptr_type, idx_type, cusparse.CUSPARSE_INDEX_BASE_ZERO, self.cuda_dtype
                )

        ##########################################################
        # Analysis Phase -- run for standard and transposed case #
        ##########################################################

        # Query buffer size needed for the analysis and execution steps
        bufferSize = cusparse.spSM_bufferSize(
                self.handle, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
                computeType=self.cuda_dtype, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
                spsmDescr=self.spsmDescr)
        bufferSizeT = cusparse.spSM_bufferSize(
                self.handle, cusparse.CUSPARSE_OPERATION_TRANSPOSE,
                cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
                computeType=self.cuda_dtype, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
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
            computeType=self.cuda_dtype, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr=self.spsmDescr, externalBuffer=self.buffer.data.ptr)

        cusparse.spSM_analysis(
            self.handle, cusparse.CUSPARSE_OPERATION_TRANSPOSE,
            cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
            alpha=self.alpha.ctypes.data, matA=self.matA, matB=self.matB, matC=self.matX,
            computeType=self.cuda_dtype, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
            spsmDescr=self.spsmDescrT, externalBuffer=self.bufferT.data.ptr)

    def __del__(self):
        if hasattr(self, 'spsmDescr'):
            cusparse.spSM_destroyDescr(self.spsmDescr)
        if hasattr(self, 'spsmDescrT'):
            cusparse.spSM_destroyDescr(self.spsmDescrT)
        if hasattr(self, 'matB'):
            cusparse.destroyDnMat(self.matB)
        if hasattr(self, 'matX'):
            cusparse.destroyDnMat(self.matX)
        if hasattr(self, 'own_matA') and self.own_matA:
            cusparse.destroySpMat(self.matA)
        if hasattr(self, '_matL0') and self._matL0 is not None:
            cusparse.destroySpMat(self._matL0)
        if hasattr(self, '_matL0t') and self._matL0t is not None:
            cusparse.destroySpMat(self._matL0t)
        if hasattr(self, 'own_handle') and self.own_handle:
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
                computeType=self.cuda_dtype, alg=cusparse.CUSPARSE_SPSM_ALG_DEFAULT,
                spsmDescr=spsmDescr, externalBuffer=buffer_ptr)

        cuda.synchronize()
        t1 = perf_counter()
        kernels.time['trsv']  += t1-t0
        kernels.calls['trsv'] += 1
        kernels.load['trsv']  += 12*self.nnz+8*(self.shape[0]+self.shape[1])
        kernels.store['trsv'] += 8*self.shape[0]
        kernels.flop['trsv'] += 2*self.nnz

    def apply_as_poly(self, b, x, k=None, transpose=False):
        '''
        Approximate L^{-1} b or L^{-T} b using a Neumann polynomial of degree k:
        L^{-1} \approx \sum_{j=0}^k (I - D^{-1}L)^j D^{-1}
        '''
        if k is None:
            k = self.poly_k
            
        if k < 0:
            return self.apply(b, x, transpose)

        if self._matL0 is None:
            raise RuntimeError("FastTrsv was not initialized for Neumann polynomial (poly_k was < 0)")

        n = self.shape[0]
        if transpose:
            # x = (I - L0t)^{-1} D^{-1} b
            kernels.vscale(self._d_inv, b, self._v_poly)
            cusparse_neumann(self._matL0t, k, self._v_poly, x, n, self.dtype, self._L0t.nnz, self.handle)
        else:
            # x = D^{-1} (I - L0)^{-1} b
            cusparse_neumann(self._matL0, k, b, self._v_poly, n, self.dtype, self._L0.nnz, self.handle)
            kernels.vscale(self._d_inv, self._v_poly, x)

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
    M^{-1} = (L+D)^{-T} D^{-1} (L+D)^{-1}.

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

