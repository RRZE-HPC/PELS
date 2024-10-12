#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

from numba import jit, prange
import numpy as np
from scipy.sparse import csr_matrix

class sellcs_matrix:

    def __init__(self, A_csr=None, A_arrays=None, C=32, sigma=1, shape=None, dtype=None):
        '''
        The sellcss_matrix is a simple object without any methods or operators. To perform a
        sparse matrix-vector multiplication, use kernels.spmv(A, x, y) instead of y=A*x.
        There are two ways to construct this matrix class:

        1. from a scipy.sparse.csr_matrix: A = sellcs_matrix(A_csr, C, sigma)
        2. from existing SELL-C-sigma arrays: A = sellcs_matrix((data,indices,indptr, permute, unpermute, nnz), C, shape)
        '''
        if (A_csr is None and A_arrays is None) or (A_csr!=None and A_arrays!=None):
            raise ValueError('Invalid input to sellcs_matrix: need either A_csr or (data, indices, indptr)')
        if C<1 or sigma<0 or (sigma!=1 and sigma%C!=0):
            raise ValueError('Invalid parameters C and/or sigma: sigma should be 1 or a multiple of C.')

        self.C = C
        self.sigma = sigma

        if A_csr != None:
            self.shape = A_csr.shape
            self.nnz = A_csr.nnz
            self.indptr, self.data, self.indices, self.permute, self.unpermute = csr_to_sellcs(A_csr.indptr, A_csr.data, A_csr.indices, C, sigma)
        elif A_arrays != None:
            self.data, self.indices, self.indptr, self.permute, self.unpermute, self.nnz = A_arrays
            self.shape = shape
        self.dtype = self.data.dtype
        self.nchunks = self.indptr.size-1
        self.nnz_stored = self.indptr[self.nchunks]

    def __str__(self):
        out = 'SELL-%d-%d matrix, shape=[%d %d]\n'%(self.C,self.sigma, self.shape[0],self.shape[1])
        if self.sigma>1:
            out += 'permute='+str(self.permute)+'\n'
            out += 'unpermute='+str(self.permute)+'\n'
        out += 'indptr='+str(self.indptr)+'\n'
        out += 'indices='+str(self.indices)+'\n'
        out += 'data='+str(self.data)+'\n'
        return out

@jit(nopython=True, parallel=True)
def csr_to_sellcs(A_indptr, A_data, A_indices, C, sigma):
    '''
    internal utility function to efficiently convert CSR matrix arrays to SELL-C-sigma format.
    Users should not call this but use sellcs_matrix(A_csr=A) instead.

    Returns: indptr, data, indices, permute, unpermute
    '''
    nrows = A_indptr.size-1
    nchunks = (nrows+C-1)//C
    indptr  = np.zeros(nchunks+1, dtype='int64')
    permute = np.arange(nrows)
    unpermute = np.empty_like(permute)
    scope = max(C,sigma)

    chunk = 0

    for row_offset in range(0,nrows, scope):
        last = min(row_offset+scope, nrows)
        # numba doesn't allow indexing with a range, so we make rng0/1 arrays for now:
        #rng0=range(row_offset, last)
        #rng1=range(row_offset+1, last+1)
        rng0=np.arange(row_offset, last)
        rng1=np.arange(row_offset+1, last+1)
        row_len = A_indptr[rng1] - A_indptr[rng0]
        # Note: the only way to make numpy sort in descending order is to
        #       give it the negative values and sort ascending :)
        if sigma>1:
            idx = np.argsort(-row_len)
            row_len = row_len[idx]
            permute[rng0] = row_offset+idx
        for i in range(row_offset, last, C):
            c = min(C, nrows-i)
            w = max(row_len[i-row_offset:i-row_offset+c])
            indptr[chunk+1] = indptr[chunk] + c*w
            chunk += 1

    nnz_stored = indptr[nchunks]
    unpermute[permute] = np.arange(nrows)
    data = np.zeros(nnz_stored, dtype=A_data.dtype)
    indices = np.zeros(nnz_stored, dtype=A_indices.dtype)

    for chunk in prange(nchunks):
        c = min(C*(chunk+1), nrows)-C*chunk
        i0 = C*chunk
        i1 = i0 + c
        for i in range(i0, i1):
            rowA = permute[i]
            k0 = A_indptr[rowA]
            k1 = A_indptr[rowA+1]
            for k in range(k0, k1):
                pos  = indptr[chunk] + (k-k0)*c + (i-i0)
                data[pos] = A_data[k]
                indices[pos] = unpermute[A_indices[k]]
    return indptr, data, indices, permute, unpermute
