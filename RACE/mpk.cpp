/*******************************************************************************************/
/* This file is part of the training material available at                                 */
/* https://github.com/jthies/PELS                                                          */
/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
/* included in this software.                                                              */
/*                                                                                         */
/* Contact: Christie Alappat (christiealappatt@gmail.com)                                  */
/*                                                                                         */
/*******************************************************************************************/



#include "sparsemat.h"
#include "densemat.h"
#include <omp.h>

extern "C"
{
void* mpk_setup(int nrows, int *rowPtr, int *col, double *val, int power, double cacheSize, bool split)
{
    sparsemat* mat = new sparsemat;
    int nnz=rowPtr[nrows];
    mat->initCover(nrows, nnz, val, rowPtr, col);
    int nthreads = 1;
#pragma omp parallel
    {
        nthreads=omp_get_num_threads();
    }
    mat->prepareForPower(power, cacheSize, nthreads);
    mat->workspace = new densemat(nrows, power);

    if(split)
    {
        mat->splitMatrixToLDU();
    }

    void* voidMat=(void*)mat;
    return voidMat;
}

int* mpk_getPerm(void* voidMat)
{
    sparsemat* mat = (sparsemat*) voidMat;
    return mat->finalPerm;
}
}

struct kernelArg
{
    sparsemat* mat;
    double* y;
    double* x;
    int power;
};

//convenience macros
#define ENCODE_TO_VOID(mat_en, x_en, y_en, power_en)\
    kernelArg *arg_encode = new kernelArg;\
    arg_encode->mat = mat_en;\
    arg_encode->y = y_en;\
    arg_encode->x = x_en;\
    arg_encode->power = power_en;\
    void* voidArg = (void*) arg_encode;\


#define DECODE_FROM_VOID(voidArg)\
    kernelArg* arg_decode = (kernelArg*) voidArg;\
    sparsemat* mat = arg_decode->mat;\
    double* y = arg_decode->y;\
    double* x = arg_decode->x;\
    int power = arg_decode->power;

#define DELETE_ARG()\
    delete arg_encode;\


inline void MPK_CALLBACK(int start, int end, int pow, int numa_domain, void* args)
{
    DECODE_FROM_VOID(args);
    for(int row=start; row<end; ++row)
    {
        double tmp = 0;
        const int offset = (pow-2)*mat->nrows;
        double *in_vec = (pow==1)?x:&(mat->workspace->val[offset]);
        const int nextOffset = (pow-1)*mat->nrows;
        double *out_vec = (pow==power)?y:&(mat->workspace->val[nextOffset]);
        //#pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)
        //#pragma nounroll
        for(int idx=mat->rowPtr[row]; idx<mat->rowPtr[row+1]; ++idx)
        {
            tmp += mat->val[idx]*in_vec[mat->col[idx]];
        }
        out_vec[row] = tmp;
    }
}

extern "C"
{
    //computes y=A^p*x
    void mpk(void* voidA, int power, double* x, double* y)
    {

        sparsemat* A = (sparsemat*)(voidA);
        RACE::Interface *ce=A->ce;
        ENCODE_TO_VOID(A, x, y, power);
        int race_power_id = ce->registerFunction(&MPK_CALLBACK, voidArg, power);
        {
            ce->executeFunction(race_power_id);
        }
        DELETE_ARG();
    }
}

inline void MPK_NEUMANN_APPLY_CALLBACK(int start, int end, int pow, int numa_domain, void* args)
{
    DECODE_FROM_VOID(args);
    int poly_k = (power-1)/2;
    if((power-1)%2 != 0)
    {
        printf("Error in MPK Neumann apply. Power value is %d and not odd\n", power);
    }
    if(pow<=poly_k)
    {
        //Compute with U
        for(int row=start; row<end; ++row)
        {
            const int offset = (pow-2)*mat->nrows;
            double *in_vec = (pow==1)?x:&(mat->workspace->val[offset]);
            const int nextOffset = (pow-1)*mat->nrows;
            double *out_vec = &(mat->workspace->val[nextOffset]);
            //#pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)
            //#pragma nounroll
            double tmp = 0;
            out_vec[row] = in_vec[row];
            for(int idx=mat->U->rowPtr[row]; idx<mat->U->rowPtr[row+1]; ++idx)
            {
                tmp -= mat->U->val[idx]*in_vec[mat->U->col[idx]];
            }
            out_vec[row] += tmp;
        }

    }
    /*else if(pow == (poly_k+1))
    {
        //Compute SpMV
        for(int row=start; row<end; ++row)
        {
            double tmp = 0;
            const int offset = (pow-2)*mat->nrows;
            double *in_vec = &(mat->workspace->val[offset]);
            const int nextOffset = (pow-1)*mat->nrows;
            double *out_vec = &(mat->workspace->val[nextOffset]);
            //#pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)
            //#pragma nounroll
            for(int idx=mat->rowPtr[row]; idx<mat->rowPtr[row+1]; ++idx)
            {
                tmp += mat->val[idx]*in_vec[mat->col[idx]];
            }
            out_vec[row] = tmp;
        }
    }*/
    //SpMV with L and U
    else if(pow == (poly_k+1))
    {
        //Compute SpMV
        for(int row=start; row<end; ++row)
        {
            double tmp = 0;
            const int offset = (pow-2)*mat->nrows;
            double *in_vec = &(mat->workspace->val[offset]);
            const int nextOffset = (pow-1)*mat->nrows;
            double *out_vec = &(mat->workspace->val[nextOffset]);
            //#pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)
            //#pragma nounroll
            for(int idx=mat->L->rowPtr[row]; idx<mat->L->rowPtr[row+1]; ++idx)
            {
                tmp += mat->L->val[idx]*in_vec[mat->L->col[idx]];
            }
            tmp += in_vec[row]; //diagonal is one
            for(int idx=mat->U->rowPtr[row]; idx<mat->U->rowPtr[row+1]; ++idx)
            {
                tmp += mat->U->val[idx]*in_vec[mat->U->col[idx]];
            }
 

            out_vec[row] = tmp;
        }
    }
    else if(pow>(poly_k+1))
    {
        //Compute with L
        for(int row=start; row<end; ++row)
        {
            const int offset = (pow-2)*mat->nrows;
            double *in_vec = &(mat->workspace->val[offset]);
            const int nextOffset = (pow-1)*mat->nrows;
            double *out_vec = (pow==power)?y:&(mat->workspace->val[nextOffset]);
            //#pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)
            //#pragma nounroll
            double tmp = 0;
            out_vec[row] = in_vec[row];
            for(int idx=mat->L->rowPtr[row]; idx<mat->L->rowPtr[row+1]; ++idx)
            {
                tmp -= mat->L->val[idx]*in_vec[mat->L->col[idx]];
            }
            out_vec[row] += tmp;
        }
    }
}

extern "C"
{
    //computes y=A^p*x
    void mpk_neumann_apply(void* voidA, int k, double* x, double* y)
    {
        sparsemat* A = (sparsemat*)(voidA);
        RACE::Interface *ce=A->ce;
        int power = 2*k+1;
        ENCODE_TO_VOID(A, x, y, power);
        int race_power_id = ce->registerFunction(&MPK_NEUMANN_APPLY_CALLBACK, voidArg, power);
        {
            ce->executeFunction(race_power_id);
        }
        DELETE_ARG();
    }
}


extern "C"
{
void mpk_free(void* voidA)
{
    sparsemat* A = (sparsemat*)(voidA);
    delete A->workspace;
    delete A;
}
}
