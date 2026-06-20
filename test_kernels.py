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
from parameterized import parameterized_class, parameterized

import os
import numpy as np
from kernels import *
import scipy


def diff_norm(x, y):
    '''
    Computes ||x - y||_2 where x and y may live either on the host 9numpy arrays)
    or device (cuda arrays)
    '''
    xh = to_host(x)
    yh = to_host(y)
    return np.linalg.norm(xh-yh)/(np.linalg.norm(xh)+np.linalg.norm(yh))


def test_detect_cudasim():
    if 'NUMBA_ENABLE_CUDASIM' in os.environ.keys():
        assert(available_gpus()>0)

def test_detect_cuda_visible_devices():
    if 'NCUDA_VISIBLE_DEVICES' in os.environ.keys():
        visible_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert(available_gpus()==len(visible_gpus))

class VectorKernelsTest(unittest.TestCase):

    def setUp(self):
        self.a=42.9
        self.b=-32
        self.n=500
        self.x_host=np.ones(self.n,dtype='float64')
        self.y_host=np.arange(self.n,dtype='float64')+1
        self.z_host=self.y_host.copy()

        # note: if there is no GPU, this just sets a pointer
        self.x = to_device(self.x_host)
        self.y = to_device(self.y_host)
        self.z = to_device(self.z_host)

        self.eps=np.finfo(np.float64).eps

    def test_init(self):
        init(self.x, self.a)
        assert(diff_norm(self.x, self.a)<self.eps)

    def test_axpby(self):
        axpby(self.a, self.x, self.b, self.z)
        assert(diff_norm(self.z, (self.a+self.b*self.y_host))<self.eps)

    def test_dot(self):
        s=dot(self.x,self.y)
        assert(abs(s-sum(self.y_host))<np.sqrt(self.eps))


@pytest.mark.parametrize('N',[30, 128, 129, 150, 1280, 2**20, 2**20+127])
def test_dot_ones(N):
    xh = np.ones(N, dtype='float64')
    yh = np.ones(N, dtype='float64')
    x = to_device(xh)
    y = to_device(yh)
    my_dot = dot(x, y)
    assert(abs(my_dot - N)<N*1e-12)

@pytest.mark.parametrize('N',[30, 128, 129, 150, 1280, 2**20, 2**20+127])
def test_dot_arange(N):
    xh = np.arange(1,N+1, dtype='float64')
    yh = 1.0/xh
    x = to_device(xh)
    y = to_device(yh)
    my_dot = dot(x, y)
    assert(abs(my_dot - N)<N*1e-12)

@pytest.mark.parametrize('N',[30, 128, 129, 150, 1280, 2**20, 2**20+127])
def test_dot_rand(N):
    np.random.seed(123456)
    xh = np.random.rand(N)
    yh = 1.0/xh
    x = to_device(xh)
    y = to_device(yh)
    my_dot = dot(x, y)
    assert(abs(my_dot - N)<N*1e-12)

@parameterized_class(('Matrix'),[
    ['Ddiag13'],
    ['Dtest33'],
    ['Dsprandn388'] ])
class SparseMatKernelsTest(unittest.TestCase):

    def setUp(self):
        self.A=scipy.sparse.csr_matrix(scipy.io.mmread('matrices/'+self.Matrix+'.mm.gz'))
        # add +1 to the diagonal because some test matrices may be singular
        self.L = scipy.sparse.tril(self.A, format='csr') + scipy.sparse.eye(self.A.shape[0])
        self.n=self.A.shape[0]
        self.x_host=np.arange(self.n, dtype='float64')+1
        self.Ax=self.A@self.x_host
        self.Lx = self.L@self.x_host
        self.Ltx = self.L.T@self.x_host
        self.y_host=np.empty_like(self.Ax)

        self.x = to_device(self.x_host)
        self.y = to_device(self.y_host)
        self.A = to_device(self.A)
        self.L = to_device(self.L)
        self.Lx = to_device(self.Lx)
        self.Ltx = to_device(self.Ltx)

        self.eps=self.n*np.finfo(np.float64).eps



    def test_sell_1_1_is_csr(self):
        Asell = sellcs.sellcs_matrix(self.A, C=1, sigma=1)
        # this should lead to excactly the same data layout as CSR:
        assert(diff_norm(self.A.data, Asell.data)==0)
        assert(diff_norm(self.A.indices, Asell.indices)==0)
        assert(diff_norm(self.A.indptr, Asell.indptr)==0)
        assert(Asell.nnz==Asell.indptr[Asell.nchunks])

    def test_sell_catch_bad_sigma(self):
        with pytest.raises(ValueError) as excinfo:
            Asell = sellcs.sellcs_matrix(self.A, C=8, sigma=25)
            assert str(excinfo.value) == "Invalid parameters  C and/or sigma: sigma should be 1 or a multilpe of C."

    def test_to_device(self):
        if available_gpus()>0:
            from numba import cuda
            assert(cuda.is_cuda_array(self.x))
            assert(cuda.is_cuda_array(self.y))

    def test_csr_spmv(self):
        spmv(self.A, self.x, self.y)
        assert(diff_norm(self.y, self.Ax) < self.eps)


    def test_csr_trsv(self):
        # solve Ly = b (=Lx) for y
        trsv(self.L, self.Lx, self.y)
        # check that y ~= x
        assert(diff_norm(self.x, self.y) < self.eps)


    def test_csr_trsv_trans(self):
        # solve L^Ty = b (=Ltx) for y
        trsv(self.L, self.Ltx, self.y, True)
        # check that y ~= x
        assert(diff_norm(self.x, self.y) < self.eps)


    def test_sell_1_1_spmv(self):
        Asell = sellcs.sellcs_matrix(self.A, C=1, sigma=1)
        if available_gpus()>0:
            Asell = to_device(Asell)
        spmv(Asell, self.x, self.y)
        assert(diff_norm(self.y, self.Ax) < self.eps)
        assert(Asell.nnz==Asell.indptr[Asell.nchunks])


    def test_sell_32_1_spmv(self):
        Asell = sellcs.sellcs_matrix(self.A, C=32, sigma=1)
        if available_gpus()>0:
            Asell = to_device(Asell)
        print(Asell.indptr)
        print(Asell.data)
        print(Asell.indices)
        assert(diff_norm(Asell.permute, np.arange(Asell.shape[0]))<self.eps)
        assert(diff_norm(Asell.unpermute, np.arange(Asell.shape[0]))<self.eps)
        spmv(Asell, self.x, self.y)
        assert(diff_norm(self.y, self.Ax) < self.eps)
        assert(Asell.nnz<=Asell.indptr[Asell.nchunks])



    def test_sell_1_5_spmv(self):
        Asell = sellcs.sellcs_matrix(self.A, C=1, sigma=5)
        y = np.empty_like(self.y)
        x = self.x_host[Asell.permute]
        Asell = to_device(Asell)
        x = to_device(x)
        y = to_device(y)
        spmv(Asell, x, y)
        y = to_host(y)
        self.y = y[Asell.unpermute]
        assert(diff_norm(self.y, self.Ax) < self.eps)

    def test_sell_32_128_spmv(self):
        Asell = sellcs.sellcs_matrix(self.A, C=32, sigma=128)
        y = np.empty_like(self.y)
        x = self.x_host[Asell.permute]
        Asell = to_device(Asell)
        x = to_device(x)
        y = to_device(y)
        spmv(Asell, x, y)
        y = to_host(y)
        self.y = y[Asell.unpermute]
        assert(diff_norm(self.y, self.Ax) < self.eps)

