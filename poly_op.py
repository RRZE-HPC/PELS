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
from scipy.sparse import *
from kernels import *
from kernels import have_RACE
import sys

class poly_op:
    '''
    Given a matrix A and an integer k, this operator constructs the splitting
    (cf. Jacobi iteration):

        D^{-1/2}AD^{-1/2} = I - (L+U)

    The 'apply' function implements a preconditioned matrix-vector product

        w = M1 A M2 v

    where M1 and M2 approximately solve

    (I-L)v=w    and    (I-U)v=w, respectively,

    using the truncated Neumann series:

        v_0 = 0
        v \approx \sum_{i=0}^k L^{i} w.

    For symmetric and positive (spd) matrices A, this preconditioned operator is spd,
    which makes it suitable for CG.

    If -use_RACE is given on the command-line, and the RACE library is found,
    cache blocking is used to increase the performance of the operator. The
    cache_size parameter can be used to fine-tune the performance with RACE.
    If RACE is not available, it is ignored.

    '''

    def __init__(self, A, poly_k, cache_size=30):
        self.k = poly_k
        self.shape = A.shape
        self.dtype = A.dtype
        # store the inverse of the square-root of the diagonal
        # s.t. isqD*A*isqD has ones on the diagonal.
        self.isqD = spdiags([1.0/np.sqrt(A.diagonal())], [0], m=self.shape[0], n=self.shape[1]).tocsr()
        self.A1 = self.isqD*A*self.isqD
        self.L = -tril(self.A1,-1).tocsr()
        self.U = -triu(self.A1,1).tocsr()
        self.mpkHandle = None
        self.use_RACE = False
        if have_RACE and '-use_RACE' in sys.argv:
            self.use_RACE = True
        self.permute =None
        self.unpermute = None
        #if we have RACE, use it
        if self.use_RACE:
            split=True
            highestPower=2*poly_k+1
            print("Using RACE for cache blocking: cache_size=", cache_size, ", power=", highestPower)
            [self.mpkHandle,self.A1]=mpk_setup(self.A1, highestPower, cache_size, split)
            self.permute=mpk_get_perm(self.mpkHandle, self.shape[0])
            self.unpermute = np.arange(self.shape[0])
            self.unpermute[self.permute] = np.arange(self.shape[0])
            #permuteute all the objects
            self.L=self.L[self.permute[:,None], self.permute]
            self.U=self.U[self.permute[:,None], self.permute]
            #work-around for diagonal, since it is not subscriptable
            #not needed diagonal is one, due to normalization
            #A_prec.isqD = spdiags([1.0/np.sqrt(A.diagonal())], [0], m=A.shape[0], n=A.shape[1])

        self.t1 = np.zeros(self.shape[0], dtype=self.dtype)
        self.t2 = np.zeros(self.shape[0], dtype=self.dtype)
        self.t3 = np.zeros(self.shape[0], dtype=self.dtype)

        # in case A has CUDA arrays, also copy over our compnents:
        self.isqD = to_device(self.isqD)
        self.A1 = to_device(self.A1)
        self.L = to_device(self.L)
        self.U = to_device(self.U)

        self.t1 = to_device(self.t1)
        self.t2 = to_device(self.t2)
        self.t3 = to_device(self.t3)

    def prec_rhs(self, b, prec_b):
        '''
        Given the right-hand side b of a linear system
        Ax=b, computes prec_b=M1\b s.t.(this op)(M2x)=M1b solves the
        equivalent preconditioned linear system for the preconditioned
        solution vector M1x
        '''
        diag_spmv(self.isqD, b, self.t1)
        self._neumann(self.L, self.k, self.t1, prec_b)

    def unprec_sol(self, prec_x, x):
        '''
        Given the right-preconditioned solution vector prec_x = M2^{-1}x,
        returns x.
        '''
        self._neumann(self.U, self.k, prec_x, x)
        diag_spmv(self.isqD, x, x)

    def apply(self, w, v):
        '''
        Apply the complete (preconditioned) operator to a vector.

           v = M1 A M2 w.

        See class description for details.
        '''
        if self.mpkHandle is None:
            self._neumann(self.U, self.k, w, self.t1)
            spmv(self.A1, self.t1, self.t2)
            self._neumann(self.L, self.k, self.t2, v)
        else:
            mpk_neumann_apply(self, w, v)

    def __del__(self):
        if hasattr(self, 'mpkHandle') and self.mpkHandle is not None:
            mpk_free(self.mpkHandle)

# protected
    def _neumann(self, M, k, rhs, sol):
        '''
        Apply truncated Neumann-series preconditioner to rhs, computing sol.

        If A = I-M, the Neumann series to approximate v=A^{-1}rhs is defined as

        inv(I-M) = sum_{k=0}^{\inf} M^k.

        It converges if ||M||<1
        (analogous to geometric series for scalar x: 1/(1-x) = sum_{k=0}^\inf x^k)
        '''
        # This is the naive implementation with 'back-to-back' spmv's.
        # Every intermediate vector M^jy is computed explicitly.

        axpby(1.0, rhs, 0.0, sol)
        for j in range(k):
            spmv(M,sol,self.t3)
            axpby(1.0,self.t3,1.0,sol)


