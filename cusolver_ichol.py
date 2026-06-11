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
import cupy
from cupy.cuda import cusolver, cusparse
import kernels
import precon

class IChol:
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
        self.handle = cusparse.create()

        # 2. Setup Sparse Matrix Descriptor (We look at the Lower triangular part)
        self.mat_descr = cusparse.createMatDescr()
        cusparse.setMatType(self.mat_descr, cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparse.setMatIndexBase(self.mat_descr, cusparse.CUSPARSE_INDEX_BASE_ZERO)

        # 3. Create Info structures for Factorization (ic0) and Solve (sv2)
        # We use single-precision or double-precision specific binding hooks
        self.csric0_info = cusparse.createCsric02Info()

        # Work arrays from the CSR matrix
        # NOTE: csric02 performs the calculation IN-PLACE, so we copy values
        self.A = A.copy()
        self.A.sort_indices()
        self.A = kernels.to_device(self.A)
        self.d_val_ptr = self.A.cu_data.__cuda_array_interface__['data'][0]
        self.d_indptr_ptr = self.A.cu_indptr.__cuda_array_interface__['data'][0]
        self.d_indices_ptr = self.A.cu_indices.__cuda_array_interface__['data'][0]

        # Memory Buffer Allocation ---
        # Determine the internal workspace required by cuSPARSE for IC and solving
        # (Using float64 / D variant hooks)

        # Factorization buffer
        buf_size_ic = cusparse.dcsric02_bufferSize(
                self.handle, m, nnz, self.mat_descr,
                self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
                self.csric0_info)


        self.pBuffer_ic    = cupy.empty(buf_size_ic, dtype=cupy.int8)

        # Analysis phase ---
        policy = cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL
        cusparse.dcsric02_analysis(
            self.handle, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csric0_info, policy, self.pBuffer_ic.data.ptr)

        # Numerical factorization
        # This alters `self.d_val` in place. After execution, self.d_val holds the L factor elements.
        cusparse.dcsric02(
            self.handle, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csric0_info, policy, self.pBuffer_ic.data.ptr)

        # Perform analysis for the subsequent triangular solves
        # Note that the A matrix we pass in is only used to get the shape and non-zero count,
        # the actual solve is done with the matrix descriptor, which holds the entries of L, not A.
        self.fast_trsv = precon.FastTrsv(self.A, cusparse_handle=self.handle, descrL=self.mat_descr)

        # allocate temporary device vector for solve phase
        self.v_tmp = cupy.empty(m, dtype=cupy.float64)


    def apply(self, b,  x):
        # --- Forward and Backward Solves (M * x = b) ---

        # Forward solve: L * v = b
        self.fast_trsv.apply(b, self.v_tmp, transpose=False)

        # Backward solve: L^T * x = v
        self.fast_trsv.apply(self.v_tmp, x, transpose=True)

    def __del__(self):
        # --- Clean up ---
        cusparse.destroyCsric02Info(self.csric0_info)
        self.fast_trsv = None

