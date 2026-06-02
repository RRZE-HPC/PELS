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

import cupy
from cupy.cuda import cusolver, cusparse
from cupyx.scipy.sparse.linalg import spilu as cupy_spilu

from kernels import *
from cuda_precon import *

from numba import cuda

# total number of calls
calls = {'setup': 0, 'apply': 0}
# total elapsed time in seconds
time = {'setup': 0, 'apply': 0.0, 'axpby': 0.0}

def invert(v):
    cu_invert.forall(v.size)(v)
    cuda.synchronize()

def vscale(v, x, y):
    '''
    Helper function to compute y = v.*x (element-wise multiplication of vectors)
    '''
    cu_vscale.forall(v.size)(v,x,y)
    cuda.synchronize()

def vscale_inplace(v, x):
    '''
    Helper function to compute x = v.*x (in-place element-wise multiplication of vectors)
    '''
    cu_vscale_inplace.forall(v.size)(v,x)
    cuda.synchronize()

def neumann(A0, k, v0, x):
    r'''
    Apply degree-k Neumann polynomial in matrix A = (I-A0):

    x = A^{-1} v0 = (I-A0)^{-1} v0 \approx \sum_{j=0}^k (A0^j v0)

    '''
    v = clone(v0)
    A0v = clone(v0)
    # set x = v0 (= A0^0 v0)
    axpby(1.0,v0, 0.0,x)
    for _ in range(k):
        # A0v = A0^{j+1}v0
        spmv(A0, v, A0v)
        # x += v0 to yield x_k
        axpby(1.0, A0v, 1.0, x)
        # swap the vectors
        A0v, v = v, A0v

class Jacobi:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        t0 = perf_counter()
        self.D_inv = to_device(A.diagonal())
        invert(self.D_inv)
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Diagonal scaling, v = D^{-1}w
        '''
        t0 = perf_counter()
        vscale(self.D_inv, w, v)
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

class SymmetricGaussSeidel:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        t0 = perf_counter()
        self.D = to_device(A.diagonal())
        self.LplusD = to_device(scipy.sparse.tril(A).tocsr())
        self.v_tmp = to_device(np.zeros(A.shape[0]))
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Symmetric Gauss-Seidel: v = (L+D)^{-T} D (L+D)^{-1} w
        '''
        t0 = perf_counter()
        trsv(self.LplusD, w, self.v_tmp)
        vscale_inplace(self.D, self.v_tmp)
        trsv(self.LplusD, self.v_tmp, v, transpose=True)
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

    def __init__(self, A, fill=1, droptol=0.0, poly_k=-1):
        '''
        Factor A \approx LL^T on the host.

        Input:

          A: scipy.sparse.csr_matrix, only host-sde is accessed.

        Output:

          self.L: lower triangular factor, copied to device using "to_device"
        '''

        t0 = perf_counter()
        self.shape = A.shape
        self.dtype = A.dtype
        # For poly_k>0 we use a Neumann polynomial to approximate the triangular solves
        self.poly_k = poly_k
        # create a temporary vector for the 'apply' function:
        self.v_tmp = to_device(np.zeros(A.shape[0]))

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
            self.L = to_device(L)
        else:
            d_inv = 1.0/d
            self.d_inv = to_device(d_inv)

            # store the (negative) factor L and its explicit transpose, but skip the diagonal:
            # L = (I - L0), L^T = (I - L0t)
            # for implementing the Neumann polynomial approximation if the inverse (see 'apply')
            L0  = scipy.sparse.tril(-L,k=-1, format='csr')
            L0t = L0.T.tocsr()
            self.L0 = to_device(L0)
            self.L0t = to_device(L0t)
            self.w_tmp = clone(self.v_tmp)

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
            trsv(self.L, w, self.v_tmp, False)
            trsv(self.L, self.v_tmp, v, True)
        else:
            # Use the degree-k Neumann polynomial to approximate the two triangular solves.
            #
            # With L = (I-L0), L^T = (I-L0t), A \approx LDL^T
            # Solve Av = (LDL^T)v = w as

            # 1. v_tmp = L^{-1}w = (I-L0)^{-1}w \approx \sum_j L0^j w
            neumann(self.L0, self.poly_k, w, self.v_tmp)

            # 2. W_tmp = D^{-1}v_tmp
            vscale(self.d_inv, self.v_tmp, self.w_tmp)

            # 3. v = L^{-T}w_tmp \approx \sum_{j=0}^k L0t^j w_tmp
            neumann(self.L0t, self.poly_k, self.w_tmp, v)

        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0


class CuPyILU:
    '''
    Zero-fill incomplete Cholesky factorization preconditioner.
    Given a symmetric and positive definite (spd) matrix A,
    computes A \approx LL^T, where L inherits the sparsity pattern
    of the lower triangular factor of A. The preconditioner is applied
    as a sequence of a forward triangular solve with L and a backward triangular
    solve with L^T.

    Example:

    A  = matrix_generator.create_matrix('Laplace128x128')
    IC = IChol(A)
    y  = kernels.to_device(numpy.random.random(A.shape[0]))
    x  = kernels.to_device(numpy.zeros(A.shape[0]))
    IC.apply(y, x)
    '''

    def __init__(self, A, fill=1, droptol=0.0):
        '''
        '''
        t0 = perf_counter()
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = as_cupy(A, as_csc=True)
        print(type(A))
        print(type(self.A))
        self.ilu = cupy_spilu(self.A) #, drop_tol=droptol, fill_factor=fill)
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0


    def apply(self, w, v):
        t0 = perf_counter()
        if not (cuda.is_cuda_array(v) and
                cuda.is_cuda_array(w)):
            raise Exception('IChol preconditioner requires vectors to be cuda arrays')

        v[:] = self.ilu.solve(as_cupy(w))
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

# cuSolver zero-fill incomplete Cholesky
# This code was generated using Google Gemini 3 and modified by J. Thies
#


class CuSolverIChol0:
    """
    Incomplete Cholesky factorization (A ~= L * L^T) implemented in CuSolver.
    A should be a cupyx.scipy.sparse.csr_matrix and should be spd.
    """
    def __init__(self, A):

        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape
        self.nnz = A.nnz

        # Extract structural dimensions
        m = A.shape[0]
        nnz = A.nnz

        # 1. Grab handles from current CuPy context
        device = cupy.cuda.Device()
        self.cusparse_handle = device.cusparse_handle
        self.cusolver_handle = device.cusolver_handle

        # 2. Setup Sparse Matrix Descriptor (We look at the Lower triangular part)
        self.mat_descr = cusparse.createMatDescr()
        cusparse.setMatType(self.mat_descr, cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparse.setMatIndexBase(self.mat_descr, cusparse.CUSPARSE_INDEX_BASE_ZERO)

        # 3. Create Info structures for Factorization (ic0) and Solve (sv2)
        # We use single-precision or double-precision specific binding hooks
        self.csric0_info = cusparse.createCsric02Info()
        self.csrsv2_info_L = cusparse.createCsrsv2Info()
        self.csrsv2_info_Lt = cusparse.createCsrsv2Info()

        # Work arrays from the CSR matrix
        # NOTE: csric02 performs the calculation IN-PLACE, so we copy values
        self.d_val = copy(A.cu_data)
        self.d_val_ptr = self.d_val.__cuda_array_interface__['data'][0]
        self.d_indptr = copy(A.cu_indptr)
        self.d_indptr_ptr = self.d_indptr.__cuda_array_interface__['data'][0]
        self.d_indices = copy(A.cu_indices)
        self.d_indices_ptr = self.d_indices.__cuda_array_interface__['data'][0]

        # Policy and trans flags
        self.policy = cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL
        self.trans_L = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        self.trans_Lt = cusparse.CUSPARSE_OPERATION_TRANSPOSE

        # --- STEP 4: Memory Buffer Allocation ---
        # Determine the internal workspace required by cuSPARSE for IC and solving
        # (Using float64 / D variant hooks)

        # Factorization buffer
        buf_size_ic = cusparse.dcsric02_bufferSize(
                self.cusparse_handle, m, nnz, self.mat_descr,
                self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
                self.csric0_info)

        # Forward solve buffer (L)
        buf_size_sv_L = cusparse.dcsrsv2_bufferSize(
            self.cusparse_handle, self.trans_L, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csrsv2_info_L)
        # Backward solve buffer (L^T)
        buf_size_sv_Lt = cusparse.dcsrsv2_bufferSize(
            self.cusparse_handle, self.trans_Lt, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr, 
            self.csrsv2_info_Lt)

        max_buf_size = max(buf_size_ic, buf_size_sv_L, buf_size_sv_Lt)
        self.pBuffer = cp.empty(max_buf_size, dtype=cp.int8)

        # --- STEP 5: Analysis phase ---
        cusparse.dcsric02_analysis(
            self.cusparse_handle, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csric0_info, self.policy, self.pBuffer.data.ptr)

        cusparse.dcsrsv2_analysis(
        self.cusparse_handle, self.trans_L, m, nnz, self.mat_descr, self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
        self.csrsv2_info_L, self.policy, self.pBuffer.data.ptr)

        cusparse.dcsrsv2_analysis(
        self.cusparse_handle, self.trans_Lt, m, nnz, self.mat_descr,
        self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
        self.csrsv2_info_Lt, self.policy, self.pBuffer.data.ptr)

        # --- Execute Incomplete Cholesky Factorization ---
        # This alters `self.d_val` in place. After execution, self.d_val holds the L factor elements.
        cusparse.dcsric02(
            self.cusparse_handle, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csric0_info, self.policy, self.pBuffer.data.ptr)

    def apply(self, b,  x):
        # --- Forward and Backward Solves (M * x = b) ---

        m = self.shape[0]
        nnz = self.nnz
        alpha = np.array(1.0, dtype=self.dtype)

        # Allocate intermediate vector
        z = cp.zeros(m, dtype=self.dtype)

        x_ptr = x.__cuda_array_interface__['data'][0]
        b_ptr = b.__cuda_array_interface__['data'][0]
        z_ptr = z.data.ptr

        # Forward solve: L * z = b
        # Tell cuSPARSE to treat the factored matrix as a Lower Triangular matrix with a non-unit diagonal
        cusparse.setMatFillMode(self.mat_descr, cusparse.CUSPARSE_FILL_MODE_LOWER)
        cusparse.setMatDiagType(self.mat_descr, cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT)

        cusparse.dcsrsv2_solve(
            self.cusparse_handle, self.trans_L, m, nnz, alpha.ctypes.data, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr, self.csrsv2_info_L,
            b_ptr, z_ptr, self.policy, self.pBuffer.data.ptr)

        # Backward solve: L^T * y = z
        cusparse.dcsrsv2_solve(
            self.cusparse_handle, self.trans_Lt, m, nnz, alpha.ctypes.data, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr, self.csrsv2_info_Lt,
            z_ptr, x_ptr, self.policy, self.pBuffer.data.ptr)
        troet_b = from_device(b)
        troet_x = from_device(x)
        troet_z = z.get() # note that this is a cupy object, not numba.cuda
        print('TROET b')
        print(troet_b)
        print('TROET z')
        print(troet_z)
        print('TROET x')
        print(troet_x)


    def __del__(self):
        # --- Clean up ---
        cusparse.destroyCsric02Info(self.csric0_info)
        cusparse.destroyCsrsv2Info(self.csrsv2_info_L)
        cusparse.destroyCsrsv2Info(self.csrsv2_info_Lt)
        cusparse.destroyMatDescr(self.mat_descr)

