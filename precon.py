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
from kernels import *

from numba import cuda

from scipy.sparse.linalg import spilu

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

def neumann(A0, k, x, y):
    '''
    Apply degree-k Neumann polynomial in matrix A = (I-A0):

    y \approx A^{-1} x = \sum_{j=0}^k A0^j x

    This function overwrites the input vector x with garbage and produces y
    '''
    v = copy(x) # = A0^0 x
    w = clone(y)
    for _ in range(k):
        spmv(A0, x, w, scalar_y=-1.0)
        axpby(1.0, v, -1.0, w)

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
    Zero-fill incomplete Cholesky factorization preconditioner.
    Given a symmetric and positive definite (spd) matrix A,
    computes A \approx LL^T, where L inherits the sparsity pattern
    of the lower triangular part of A. The preconditioner is applied
    as a sequence of a forward triangular solve with L and a backward triangular
    solve with L^T.

    Implementation note: The factorization is done on the host (CPU) using scipy, and in fact
    uses an ILU algorithm without pivoting, so the 'setup' phase is highly non-optimal.
    The triangular solves rely on the CuPy implementation (via trsv in kernels.py).
    If poly_k>=0 is given, triangular solves are replaced by a degree <poly_k> Neumann polynomial,
    resulting in 2x <poly_k> spmvs with triangular matrices instead of two triangular solves.

    Example:

    A  = matrix_generator.create_matrix('Laplace128x128')
    IC = IChol(A)
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
        ilu = spilu(A, drop_tol=droptol, fill_factor=fill, permc_spec='NATURAL', diag_pivot_thresh=0.0)

        L = ilu.L
        d = np.sqrt(ilu.U.diagonal())
        D = np.diag(d)

        # scale such that A \approx LL^T
        L = (L@D).to_csr()

        if poly_k<0:
            # copy to GPU for subsequent triangular solves
            self.L = to_device(L)
        else:
            d_inv = 1.0/d
            # scale U such that L = U^T: U <- 1/sqrt(d)*U.
            # We do this to avoid having to implement spmv with the transposed matrix L^T.
            Lt = (np.diag(d_inv) @ ilu.U).tocsr() # now Lt = L^T
            self.d_inv = to_device(d_inv)

            # store the (negative) factor L and its explicit transpose, but skip the diagonal:
            # L = (I - L0), L^T = (I - L0t)
            # for implementing the Neumann polynomial approximation if the inverse (see 'apply')
            self.L0  = to_device(scipy.sparse.tril(-L,k=-1, format='csr'))
            self.L0t = to_device(scipy.sparse.triu(-Lt,k=+1, format='csr'))

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
            # With L = D(I-L0), L^T = (I-L0t)D
            # Solve Av = (LL^T)v = w as

            # 1. v_tmp = L^{-1}w \approx \sum_{j=0}^k (I-L0)^j (D^{-1}v)
            vscale(self.d_inv, w, w_tmp)
            neumann(self.L0, self.w_tmp. self.v_tmp)

            # 2. v = L^{-T}v_tmp \approx \sum_{j=0}^k (I-L0)^j D^{-1}v_tmp
            # This again overwrites v_tmp and produces w_tmp
            neumann(self.L0t, self.v_tmp. self.w_tmp)
            vscale(self.d_inv, w_tmp, v)
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
        self.A = to_device(scipy.sparse.tril(A).tocsr())
        self.ilu = spilu(as_cupy(A), drop_tol=droptol, fill_factor=fill)
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
