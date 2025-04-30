
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import spsolve_triangular
from kernels import *
from kernels import have_RACE
import sys
from numpy.linalg import norm

class ilu_op:
    '''
    Given a matrix A and an integer k, this operator constructs ILU preconditioner:
    Solving system would be
    L{-1} A U{-1} U x = L{-1} b

    The 'apply' function implements a preconditioned matrix-vector product
        w = L{-1} A U{-1} v
        
    The 'prec_rhs' routine computes
        w = L{-1}v
    
    The 'unprec_sol' routine computes
        w = U{-1}v

    For symmetric and positive (spd) matrices A, this preconditioned operator is spd,
    since L^{T}=U which makes it suitable for CG.
    '''

    def __init__(self, A, makeSymm=False):
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = A
        self.ilu0 = ilu0_setup(A)
        self.makeSymm = makeSymm
        self.L = tril(self.ilu0,-1).tocsr() + eye(self.shape[0])
        self.U = triu(self.ilu0,0).tocsr()
        if makeSymm:
            #extract D from U to make LDU factorization
            D = self.U.diagonal()
            #scale rows in U according to D to make unit diagonal
            self.U = spdiags([1.0/D], [0], m=self.shape[0], n=self.shape[1])*self.U
            #now scale both L and U by sqrt(D)
            D_sqrt = spdiags([np.sqrt(D)], [0], m=self.shape[0], n=self.shape[1]).tocsr()
            self.L = self.L*D_sqrt
            self.U = D_sqrt*self.U
        
        self.t1 = np.zeros(self.shape[0], dtype=self.dtype)
        self.t2 = np.zeros(self.shape[0], dtype=self.dtype)
        #setup TRSV for solving
        self.L_solve_handle = trsv_setup(1,self.L)
        self.U_solve_handle = trsv_setup(0,self.U)
        # in case A has CUDA arrays, also copy over our compnents:
        self.L = to_device(self.L)
        self.U = to_device(self.U)
        self.A = to_device(self.A)
        self.t1 = to_device(self.t1)
        self.t2 = to_device(self.t2)

    def prec_rhs(self, b, prec_b):
        '''
        Given the right-hand side b of a linear system
        Ax=b, computes prec_b = L^{-1}b
        '''
        #self.t1 = spsolve_triangular(self.L, b, lower=True)
        trsv(1,self.L_solve_handle,b,self.t1)
        axpby(1.0, self.t1, 0.0, prec_b)


    def unprec_sol(self, prec_x, x):
        '''
        Given the right-preconditioned solution vector prec_x = U^{-1}x
        returns x.
        '''
        #self.t1 = spsolve_triangular(self.U, prec_x, lower=False)
        trsv(0, self.U_solve_handle,prec_x,self.t1)
        axpby(1.0, self.t1, 0.0, x)

    def apply(self, w, v):
        '''
        Apply the complete (preconditioned) operator to a vector.

           v = L^{-1} A U{-1} w.

        See class description for details.
        '''
        #self.t1 = spsolve_triangular(self.U, w, lower=False)
        trsv(0, self.U_solve_handle,w,self.t1)
        spmv(self.A, self.t1, self.t2)
        #self.t1 = spsolve_triangular(self.L, self.t2, lower=True)
        trsv(1,self.L_solve_handle,self.t2,self.t1)
        axpby(1.0, self.t1, 0.0, v)

    def __del__(self):
        #destroy TRSV handles
        if(self.L_solve_handle != None):
            trsv_free(self.L_solve_handle)
        if(self.U_solve_handle != None):
            trsv_free(self.U_solve_handle)
        


