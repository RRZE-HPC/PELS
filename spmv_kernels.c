/*******************************************************************************************/
/* This file is part of the training material available at                                 */
/* https://github.com/jthies/PELS                                                          */
/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
/* included in this software.                                                              */
/*                                                                                         */
/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
/*                                                                                         */
/*******************************************************************************************/

#include <sys/types.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Operation that computes y=Ax.
void csr_spmv(size_t N, const double *restrict val, const int *restrict rowPtr, const int *restrict col, const double *restrict x, double *restrict y)
{
#pragma omp parallel for schedule(runtime)
    for(int row=0; row<N; ++row)
    {
        double tmp=0;
        for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
        {
            tmp += val[idx]*x[col[idx]];
        }
        y[row]=tmp;
    }
}

// function that copies the data from CSR (or SELL-C-sigma) arrays to new ones
// using the same OpenMP scheduling that would be used during spmv.
void copy_csr_arrays(size_t N,
        const double *restrict Aval, const int *restrict ArowPtr, const int *restrict Acol,
              double *restrict val,        int *restrict rowPtr,        int *restrict col)
{
#pragma omp parallel for schedule(runtime)
    for(int row=0; row<N; ++row)
    {
        rowPtr[row]=ArowPtr[row];
        rowPtr[row+1]=ArowPtr[row+1];
        for(int idx=ArowPtr[row]; idx<ArowPtr[row+1]; ++idx)
        {
            val[idx] += Aval[idx];
            col[idx]=Acol[idx];
        }
    }
}


//symmetrically permutes the CSR matrix
void permute_csr_arrays(int *perm, size_t nrows,
        const double *restrict src_val, const int *restrict src_rowPtr, const int *restrict src_col,
              double *restrict dest_val,        int *restrict dest_rowPtr,        int *restrict dest_col
)
{

    int *invPerm = NULL;
    
    if(perm)
    {
        invPerm = (int*)malloc(sizeof(int)*nrows);
    #pragma omp parallel for schedule(static)
        for(int i=0; i<nrows; ++i) {
            invPerm[perm[i]] = i;
        }
    }
    
    dest_rowPtr[0] = 0;
    //NUMA init
#pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
    {
        dest_rowPtr[row+1] = 0;
    }


    if(perm != NULL)
    {
        //first find dest_rowPtr; therefore we can do proper NUMA init
        int permIdx=0;
        for(int row=0; row<nrows; ++row)
        {
            //row permutation
            int permRow = perm[row];
            for(int idx=src_rowPtr[permRow]; idx<src_rowPtr[permRow+1]; ++idx)
            {
                ++permIdx;
            }
            dest_rowPtr[row+1] = permIdx;
        }
    }
    else
    {
        for(int row=0; row<nrows+1; ++row)
        {
            dest_rowPtr[row] = src_rowPtr[row];
        }
    }

    if(perm != NULL)
    {
        //with NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            //row permutation
            int permRow = perm[row];
            for(int permIdx=dest_rowPtr[row],idx=src_rowPtr[permRow]; permIdx<dest_rowPtr[row+1]; ++idx,++permIdx)
            {
                //permute column-wise also
                //dest_val[permIdx] = val[idx];
                dest_col[permIdx] = invPerm[src_col[idx]];
                dest_val[permIdx] = src_val[idx];
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            for(int idx=dest_rowPtr[row]; idx<dest_rowPtr[row+1]; ++idx)
            {
                //dest_val[idx] = val[idx];
                dest_col[idx] = src_col[idx];
                dest_val[idx] = src_val[idx];
            }
        }
    }
    
    if(invPerm)
    {
        free(invPerm);
    }

}

