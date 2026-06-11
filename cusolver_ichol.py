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
import cupy as cp
from cupy.cuda import cusolver, cusparse
import kernels
import precon

class IChol:
    '''
    Incomplete Cholesky factorization (A ~= L * L^T) implemented in cuSPARSE.
    A should be a scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix and should be spd.
    '''
    def __init__(self, A):

        self.dtype = A.dtype
        self.shape = A.shape
        self.nnz = A.nnz
        m = A.shape[0]
        nnz = A.nnz

        # 1. Select precision-specific functions
        if self.dtype == np.float32:
            bufferSize_func = cusparse.scsric02_bufferSize
            analysis_func   = cusparse.scsric02_analysis
            factor_func     = cusparse.scsric02
            pivot_func      = cusparse.scsric02_zeroPivot
        elif self.dtype == np.float64:
            bufferSize_func = cusparse.dcsric02_bufferSize
            analysis_func   = cusparse.dcsric02_analysis
            factor_func     = cusparse.dcsric02
            pivot_func      = cusparse.xcsric02_zeroPivot
        else:
            raise TypeError(f"Unsupported dtype {self.dtype}. Expected float32 or float64.")

        # 2. Grab handle from current CuPy context
        self.handle = cusparse.create()

        # 3. Setup Legacy Sparse Matrix Descriptor
        self.mat_descr = cusparse.createMatDescr()
        cusparse.setMatType(self.mat_descr, cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparse.setMatIndexBase(self.mat_descr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
        cusparse.setMatFillMode(self.mat_descr, cusparse.CUSPARSE_FILL_MODE_LOWER)

        # 4. Create Info structure for Factorization
        self.csric0_info = cusparse.createCsric02Info()

        # 5. Prepare work arrays (IN-PLACE factorization)
        # We copy A to avoid modifying the original matrix.
        self.A = A.copy()
        self.A.sort_indices()
        
        # Ensure matrix is on device and get pointers
        if hasattr(self.A, 'data') and hasattr(self.A.data, 'data') and hasattr(self.A.data.data, 'ptr'):
            # Already a CuPy sparse matrix
            self.d_val_ptr = self.A.data.data.ptr
            self.d_indptr_ptr = self.A.indptr.data.ptr
            self.d_indices_ptr = self.A.indices.data.ptr
        else:
            # Fallback to kernels.to_device (attaches Numba device arrays)
            self.A = kernels.to_device(self.A)
            self.d_val_ptr = self.A.cu_data.__cuda_array_interface__['data'][0]
            self.d_indptr_ptr = self.A.cu_indptr.__cuda_array_interface__['data'][0]
            self.d_indices_ptr = self.A.cu_indices.__cuda_array_interface__['data'][0]

        # 6. Memory Buffer Allocation
        buf_size_ic = bufferSize_func(
                self.handle, m, nnz, self.mat_descr,
                self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
                self.csric0_info)

        self.pBuffer_ic = cp.empty(buf_size_ic, dtype=cp.int8)

        # 7. Analysis phase
        policy = cusparse.CUSPARSE_SOLVE_POLICY_USE_LEVEL
        analysis_func(
            self.handle, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csric0_info, policy, self.pBuffer_ic.data.ptr)

        # 8. Numerical factorization
        # This alters `self.A` data in place.
        factor_func(
            self.handle, m, nnz, self.mat_descr,
            self.d_val_ptr, self.d_indptr_ptr, self.d_indices_ptr,
            self.csric0_info, policy, self.pBuffer_ic.data.ptr)

        # Check for zero pivot
        # The position is an int32 in the underlying C API and it MUST be a host pointer
        # if the handle is in the default host pointer mode.
        position = np.zeros(1, dtype=np.int32)
        pivot_func(self.handle, self.csric0_info, position.ctypes.data)
        
        zero_pivot = int(position[0])
        if zero_pivot >= 0:
            raise RuntimeError(f"Numerical factorization failed: zero pivot found at row {zero_pivot}")

        # 9. Perform analysis for the subsequent triangular solves
        # We let FastTrsv manage its own descriptors to avoid mixing legacy/generic APIs.
        self.fast_trsv = precon.FastTrsv(self.A, cusparse_handle=self.handle)

        # allocate temporary device vector for solve phase
        self.v_tmp = cp.empty(m, dtype=self.dtype)


    def apply(self, b, x):
        # --- Forward and Backward Solves (M * x = b) ---

        # Forward solve: L * v = b
        self.fast_trsv.apply(b, self.v_tmp, transpose=False)

        # Backward solve: L^T * x = v
        self.fast_trsv.apply(self.v_tmp, x, transpose=True)

    def __del__(self):
        # --- Clean up ---
        if hasattr(self, 'csric0_info'):
            cusparse.destroyCsric02Info(self.csric0_info)
        if hasattr(self, 'mat_descr'):
            cusparse.destroyMatDescr(self.mat_descr)
        self.fast_trsv = None # FastTrsv handles its own cleanup
        if hasattr(self, 'handle'):
            cusparse.destroy(self.handle)

