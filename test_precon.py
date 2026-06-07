#/*******************************************************************************************/IChol
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
from matrix_generator import create_matrix
from precon import *
from test_kernels import diff_norm

import scipy.sparse as sp

def check_same_sparsity_pattern(A1, A2):
    """
    Checks if the sparsity pattern of A1 and A2 are the same
    """
    # Check dimensions first
    assert A1.shape == A2.shape

    # Make sure both are CSR matrices
    if not sp.issparse(A1) or A1.format != 'csr':
        A1_csr = A1.tocsr()
    else:
        A1_csr = A1
    if not sp.issparse(A2) or A2.format != 'csr':
        A2_csr = A2.tocsr()
    else:
        A2_csr = A2

    # Eliminate explicit zeros and sum duplicates
    # to ensure we are comparing actual structural non-zeros.
    A1_csr.sum_duplicates()
    A1_csr.eliminate_zeros()
    A2_csr.sum_duplicates()
    A2_csr.eliminate_zeros()

    # Compare the CSR structural arrays
    # indptr checks if rows have the same number of non-zeros
    # indices checks if those non-zeros are in the same columns
    same_row_structure = np.array_equal(A1_csr.indptr, A2_csr.indptr)
    same_col_structure = np.array_equal(A1_csr.indices, A2_csr.indices)

    if not (same_row_structure and same_col_structure):
        if A1.shape[0] <= 20:
            print(A1_csr.todense())
            print(A2_csr.todense())
        else:
            print(A1_csr)
            print(A2_csr)

    assert same_row_structure
    assert same_col_structure



@parameterized_class(('matrix', 'fill', 'droptol'),[
    ['Laplace10x10', 1, 0.0],
    ['Laplace10x10', 2, 0.0],
    ['Laplace10x10', 2, 0.01]
    ])
class ICholTest(unittest.TestCase):

    def setUp(self):
        self.matrix = 'Laplace4x4'
        self.A = create_matrix(self.matrix)
        self.ic         = IChol(self.A, droptol=0.0, fill=1, poly_k=-1)
        self.ic_fast    = IChol(self.A, droptol=0.0, fill=1, poly_k=-1, fast_trsv=True)
        self.ic_poly    = IChol(self.A, droptol=0.0, fill=1, poly_k=1)
        self.tol = 1.0e-10


#    def test_pattern_IC1(self):
#        # check that the pattern of L is the same as the lower triangular part of A
#        check_same_sparsity_pattern(sp.tril(self.A@self.A), self.ic.L)
#
#    def test_IChol_poly_split(self):
#        # reconstruct the aproximated matrix LL^T resp. (I-L0)D(I-L0)^T -- they should be the same
#        n = self.A.shape[0]
#        I = sp.eye(n)
#        D  = sp.diags(1.0/from_device(self.ic_poly.d_inv))
#        Lp = (I-from_device(self.ic_poly.L0))
#        Lpt = (I-from_device(self.ic_poly.L0t))
#
#        # check sparsity patterns
#        check_same_sparsity_pattern(Lp, self.ic.L)
#        check_same_sparsity_pattern(Lpt, self.ic.L.T)
#        # check values
#        assert(sp.linalg.norm(Lp.T - Lpt, np.inf)==0)
#        assert(sp.linalg.norm(self.ic.L@self.ic.L.T - Lp@D@Lpt, np.inf) < sp.linalg.norm(self.A, np.inf) * 1e-14)
#        # make sure all trsvs are counted
#        assert(kernels.calls['trsv'] == 2)

    def test_fast_trsv(self):
        n = self.A.shape[0]
        x_ex = to_device(np.random.random((n,)))
        x1    = clone(x_ex)
        x2    = clone(x_ex)
        b    = clone(x_ex)
        spmv(self.ic.L, x_ex, b)
        reset_counters()
        trsv(self.ic.L, b, x1, transpose=False)
        self.ic_fast.fast_trsv.apply(b, x2, transpose=False)
        assert(diff_norm(x1, x_ex)<self.tol)
        assert(diff_norm(x2, x_ex)<self.tol)

    def test_fast_trsvT(self):
        n = self.A.shape[0]
        x_ex = to_device(np.random.random((n,)))
        b    = to_device(self.ic.L.T @ x_ex)
        x1    = clone(x_ex)
        x2    = clone(x_ex)

        reset_counters()
        trsv(self.ic.L, b, x1, transpose=True)
        self.ic_fast.fast_trsv.apply(b, x2, transpose=True)
        assert(diff_norm(x1, x_ex)<self.tol)
        assert(diff_norm(x2, x_ex)<self.tol)
        # make sure all trsvs are counted
        assert(kernels.calls['trsv'] == 2)

