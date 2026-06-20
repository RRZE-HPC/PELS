import math
import numpy as np
import numba
from numba import cuda, float64

@cuda.jit(device=True)
def d_swap(arr,i,j):
    '''
    Device function to swap the contents of arr[i] and arr[j]
    '''
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp

@cuda.jit(device=True)
def d_sort(val, idx, i0, i1):
    '''
    sort arrays (val[i0:i1], idx[i0:i1]) according to idx[i0:i1].
    '''
    for i in range(0,i1-i0):
        for j in range(i0, i1 - i - 1):
            if idx[j] > idx[j + 1]:
                d_swap(val, j, j + 1)
                d_swap(idx, j, j + 1)

@cuda.jit
def cu_invert(v: float64[:]):
    '''
    given a vector v, overwite v[i] = 1.0/v[i]
    '''
    idx = cuda.grid(1)
    if idx < v.size:
        v[idx] = 1.0/v[idx]

@cuda.jit
def cu_ichol_pre(data, indptr, row_ind, col_ind):
    '''
    Given a CSR matrix on the GPU, makes sure that
    the column indices are sorted in every row.
    If the matrix is lower triangular, this makes sure
    that the diagonal element is stoed as the last non-zero.
    Simultaneously, places the row index of every element in row_ind,
    s.t. (data,row_inds,col_ind) is a COO representation of A.

    This is used as a preprocessing step for the triangular matrix that
    goes into cu_ichol0 below.
    '''
    row = cuda.grid(1)
    if row>=indptr.size:
        return

    first  = indptr[row]
    last   = indptr[row+1]
    for idx in range(first,last):
        row_ind[idx] = row
    d_sort(data, col_ind, first, last)

@cuda.jit
def cu_ichol0(L_data, L_row_idx, L_col_idx, L_indptr):
    '''
    CUDA implementation of an asynchronous incomplete Cholesky-0
    factorization.

    Input: L=tril(A).tocsr(), with (L_data, L_col_idx, L_indptr) the CSR arrays (obtained from to_device(L)),
           and (L_row_idx, L_col_idx) the (i,j) arrays, obtained from to_device(L.tocoo().[row,col])
    This kernels is to be launched with at least L.nnz threads.
    '''
    # Map each thread to an element in the sparse data array
    idx = cuda.grid(1)

    if idx >= L_data.size:
        return

    # Get the original value from matrix A before overwriting it.
    # We need it for our fix-point iteration below.
    aij = L_data[idx]

    # row and column index (i,j) for this thread:
    row_i = L_row_idx[idx]
    col_j = L_col_idx[idx]

    is_diag = (row_i==col_j)

    # start and end of rows i and j
    start_i = L_indptr[row_i]
    end_i = L_indptr[row_i + 1]
    start_j = L_indptr[col_j]
    end_j = L_indptr[col_j + 1]
    # We enforce beforehnd that the diagonal entry of row j is in indptr[j+1]-1
    # (the last element of row j) using the cu_diag_last kernel above
    diag_j = end_j-1

    # Asynchronous iteration: Based on the current L,
    # compute sum_k(L_ik * L_jk). The elements L_ik, Ljk)
    # may be updated by other threads while we're at it, but
    # the method should converge for spd matrices.
    # We allow a maximum number of 10 sweeps per element,
    # and stop if the update becomes very small
    maxit = 20
    convtol = abs(aij)*1e-8
    for _ in range(maxit):

        sum_lk = 0.0

        # 'sparse dot product':
        curr_i = start_i
        curr_j = start_j

        while curr_i < end_i and curr_j < diag_j:
            k_i = L_col_idx[curr_i]
            k_j = L_col_idx[curr_j]

            # We can exploit that i>=j (we're only working on a lower triangular matrix).
            # The j-loop never reaches diag_j, so k_i<=k_j<col_j
            if k_i >= col_j:
                break
            if k_i == k_j:
                sum_lk += L_data[curr_i] * L_data[curr_j]
                curr_i += 1
                curr_j += 1
            elif k_i < k_j:
                curr_i += 1
            else:
                curr_j += 1

        # Update L_data[idx] based on whether it is a diagonal or off-diagonal
        L_ij_old = L_data[idx]
        if is_diag:
            # Diagonal: L_ii = sqrt(a_ii - sum)
            lii = aij - sum_lk
            if lii>=0:
                L_data[idx] = math.sqrt(lii)
        else:
            # Off-diagonal: L_ij = (a_ij - sum) / L_jj
            L_data[idx] = (aij - sum_lk) / L_data[diag_j]

        #if abs(L_ij_old-L_data[idx])<convtol:
        #    break
