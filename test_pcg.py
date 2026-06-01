#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import unittest
import pytest
from parameterized import parameterized_class

import numpy as np
from kernels import *
from pcg import *
from matrix_generator import create_matrix
from test_kernels import diff_norm


def create_precon(A_csr, label):

    M = None
    # default options (IC0)
    ilu_droptol = 0.0
    ilu_fill = 1
    ilu_poly = -1
    # setup preconditioner...
    if   label == 'Jacobi' or label == 'jacobi':
        M = precon.Jacobi(A_csr)
    elif label == 'SGS':
        M = precon.SymmetricGaussSeidel(A_csr)
    elif label == 'SciPyIC0':
        M = precon.IChol(A_csr, ilu_fill, ilu_droptol, ilu_poly)
    elif label=='CuSolverIC0':
            M = precon.CuSolverIChol0(A_csr)
    elif label is not None:
            raise(f'precon label not implemented in test_pcg.py: {label}')
    return M

@parameterized_class(('precon', 'matrix', 'maxit'),[
    [None, 'Laplace10x10', 30],
    ['Jacobi','Laplace10x10', 30],
    ['SGS','Laplace10x10', 20],
    ['SciPyIC0','Laplace10x10', 25],
    ['CuSolverIC0', 'Laplace10x10', 25],
    [None, 'Laplace20x20', 60],
    ['Jacobi','Laplace40x40',120],
    ['SGS','Laplace40x40', 90],
    ['SciPyIC0','Laplace40x40', 66],
    ['CuSolverIC0','Laplace40x40', 66] ])
class CgTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345678)
        self.tol = 1.0e-6
        if self.matrix.endswith('.mm'):
            self.A = scipy.io.mmread(self.Matrix).tocsr()
        else:
            self.A = create_matrix(self.matrix)
        self.x_ex =np.random.rand(self.A.shape[0])
        self.b = self.A*self.x_ex
        self.x0 = np.zeros(self.A.shape[0], self.A.dtype)
        self.r = np.empty_like(self.x0)

        if available_gpus()>0:
            self.A = to_device(self.A)
            self.x0 = to_device(self.x0)
            self.b = to_device(self.b)
            self.r = to_device(self.r)
        self.norm_b = np.sqrt(dot(self.b,self.b))
        self.M = create_precon(self.A, self.precon)

    def test_precon_spd(self):
        '''
        Tests that for a random tall-skinny matrix V, V^TMV is symmetric and has positive diagonal entries.
        '''
        n = self.A.shape[0]
        k = min(n, 20)
        V = np.random.random((n,k)).copy(order='F')
        W = np.zeros((n,k),order='F')
        for j in range(k):
            vj = to_device(V[:,j])
            wj = to_device(W[:,j])
            if self.M is not None:
                self.M.apply(vj, wj)
            else:
                wj = copy(vj)
            wj_h = from_device(wj)
            assert(not any(np.isnan(wj_h)))
            assert(not any(np.isinf(wj_h)))
            W[:,j] = from_device(wj)
        VMV = V.T @ W

        assert(np.linalg.norm(VMV.T - VMV, np.inf)<self.tol)
        assert(min(VMV.diagonal())>0.01)

    def test_pcg(self):

        rhs = self.b
        A = self.A

        sol, relres, iter = cg_solve(A, self.M, rhs, self.x0, self.tol, self.maxit)

        x = copy(sol)

        assert(diff_norm(x, self.x_ex)/self.norm_b<self.tol)
        r = copy(x)
        spmv(self.A, x, r)
        axpby(1.0,self.b, -1.0, r)
        norm_r = np.sqrt(dot(r,r))
        assert(norm_r/self.norm_b<self.tol)
