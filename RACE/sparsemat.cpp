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
#include "stdlib.h"
#include <omp.h>
#include <vector>
#include <sys/mman.h>
#include "config_eg.h"
#include <mkl.h>
#ifdef RACE_USE_SPMP
    #include "SpMP/CSR.hpp"
    #include "SpMP/reordering/BFSBipartite.hpp"
#endif
#include "timer.h"
#include "kernels.h"
#include "densemat.h"

sparsemat::sparsemat():nrows(0), nnz(0), ce(NULL), val(NULL), rowPtr(NULL), col(NULL), nnz_symm(0), rowPtr_symm(NULL), col_symm(NULL), val_symm(NULL), diagFirst(false), colorType("RACE"), colorBlockSize(64), colorDist(-1), ncolors(-1), colorPtr(NULL), partPtr(NULL), block_size(1), rcmInvPerm(NULL), rcmPerm(NULL), finalPerm(NULL), finalInvPerm(NULL), symm_hint(false), L(NULL), U(NULL), D(NULL)
{
}

//to transfer from a different library the data structure
//need to be called after contructor
void sparsemat::initCover(int nrows_, int nnz_, double *val_, int *rowPtr_, int *col_)
{
    nrows=nrows_;
    nnz=nnz_;
    val=val_;
    rowPtr=rowPtr_;
    col=col_;
}

//performs deep copy of basic data structure
void sparsemat::basicDeepCopy(sparsemat *otherMat)
{
    nrows=otherMat->nrows;
    nnz=otherMat->nnz;

    rowPtr = new int[nrows+1];
    col = new int[nnz];
    val = new double[nnz];

    rowPtr[0] = otherMat->rowPtr[0];
#pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
    {
        rowPtr[row+1] = otherMat->rowPtr[row+1];
        for(int idx=otherMat->rowPtr[row]; idx<otherMat->rowPtr[row+1]; ++idx)
        {
            val[idx] = otherMat->val[idx];
            col[idx] = otherMat->col[idx];
        }
    }
}


sparsemat::~sparsemat()
{
    if(val)
        delete[] val;

    if(rowPtr)
        delete[] rowPtr;

    if(col)
        delete[] col;

    if(val_symm)
        delete[] val_symm;

    if(rowPtr_symm)
        delete[] rowPtr_symm;

    if(col_symm)
        delete[] col_symm;

    if(ce)
        delete ce;

    if(rcmPerm)
        delete[] rcmPerm;

    if(rcmInvPerm)
        delete[] rcmInvPerm;

    if(finalPerm)
        delete[] finalPerm;

    if(finalInvPerm)
        delete[] finalInvPerm;

/*    if(rowPtr_bcsr)
        delete[] rowPtr_bcsr;

    if(col_bcsr)
        delete[] col_bcsr;

    if(val_bcsr)
        delete[] val_bcsr;*/

    if(L)
        delete L;

    if(U)
        delete U;

    if(D)
        delete D;
}

void sparsemat::printTree()
{
    ce->printZoneTree();
}


bool sparsemat::isAnyDiagZero()
{
    //check whether a new allocation is necessary
    int extra_nnz=0;
    for(int row=0; row<nrows; ++row)
    {
        bool diagHit = false;
        for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
        {
            if(col[idx] == row)
            {
                if(val[idx] != 0)
                {
                    diagHit = true;
                }
            }
        }
        if(!diagHit)
        {
            return true;
        }
    }
    return false;
}

//necessary for GS like kernels
void sparsemat::makeDiagFirst(double missingDiag_value, bool rewriteAllDiag_with_maxRowSum)
{
    double maxRowSum=0.0;
    if(!diagFirst || rewriteAllDiag_with_maxRowSum)
    {
        //check whether a new allocation is necessary
        int extra_nnz=0;
        std::vector<double>* val_with_diag = new std::vector<double>();
        std::vector<int>* col_with_diag = new std::vector<int>();
        std::vector<int>* rowPtr_with_diag = new std::vector<int>(rowPtr, rowPtr+nrows+1);

        for(int row=0; row<nrows; ++row)
        {
            bool diagHit = false;
            double rowSum=0;
            for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
            {
                val_with_diag->push_back(val[idx]);
                col_with_diag->push_back(col[idx]);
                rowSum += val[idx];

                if(col[idx] == row)
                {
                    diagHit = true;
                }
            }
            if(!diagHit)
            {
                val_with_diag->push_back(missingDiag_value);
                col_with_diag->push_back(row);
                ++extra_nnz;
                rowSum += missingDiag_value;
            }
            maxRowSum = std::max(maxRowSum, std::abs(rowSum));
            rowPtr_with_diag->at(row+1) = rowPtr_with_diag->at(row+1) + extra_nnz;
        }

        //allocate new matrix if necessary
        if(extra_nnz)
        {
            delete[] val;
            delete[] col;
            delete[] rowPtr;

            nnz += extra_nnz;
            val = new double[nnz];
            col = new int[nnz];
            rowPtr = new int[nrows+1];

            rowPtr[0] = rowPtr_with_diag->at(0);
#pragma omp parallel for schedule(static)
            for(int row=0; row<nrows; ++row)
            {
                rowPtr[row+1] = rowPtr_with_diag->at(row+1);
                for(int idx=rowPtr_with_diag->at(row); idx<rowPtr_with_diag->at(row+1); ++idx)
                {
                    val[idx] = val_with_diag->at(idx);
                    col[idx] = col_with_diag->at(idx);
                }
            }
            printf("Explicit 0 in diagonal entries added\n");
        }

        delete val_with_diag;
        delete col_with_diag;
        delete rowPtr_with_diag;

#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            bool diag_hit = false;

            double* newVal = new double[rowPtr[row+1]-rowPtr[row]];
            int* newCol = new int[rowPtr[row+1]-rowPtr[row]];
            for(int idx=rowPtr[row], locIdx=0; idx<rowPtr[row+1]; ++idx, ++locIdx)
            {
                //shift all elements+1 until diag entry
                if(col[idx] == row)
                {
                    if(rewriteAllDiag_with_maxRowSum)
                    {
                        newVal[0] = maxRowSum;
                    }
                    else
                    {
                        newVal[0] = val[idx];
                    }
                    newCol[0] = col[idx];
                    diag_hit = true;
                }
                else if(!diag_hit)
                {
                    newVal[locIdx+1] = val[idx];
                    newCol[locIdx+1] = col[idx];
                }
                else
                {
                    newVal[locIdx] = val[idx];
                    newCol[locIdx] = col[idx];
                }
            }
            //assign new Val
            for(int idx = rowPtr[row], locIdx=0; idx<rowPtr[row+1]; ++idx, ++locIdx)
            {
                val[idx] = newVal[locIdx];
                col[idx] = newCol[locIdx];
            }

            delete[] newVal;
            delete[] newCol;
        }
        diagFirst = true;
    }
}

bool sparsemat::computeSymmData()
{
    //this is assumed by SymmSpMV kernel and GS
    makeDiagFirst();

    //compute only if previously not computed
    if(nnz_symm == 0)
    {
        /* Here we compute symmetric data of matrix
         * which is used if necessary; upper symmetric
         * portion is stored*/

        nnz_symm = 0;
        rowPtr_symm = new int[nrows+1];
        rowPtr_symm[0] = 0;

        //NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            rowPtr_symm[row+1] = 0;
        }

        //count non-zeros in upper-symm
        for(int row=0; row<nrows; ++row) {
            for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx) {
                if(col[idx]>=row) {
                    ++nnz_symm;
                }
                rowPtr_symm[row+1] = nnz_symm;
            }
        }

        col_symm = new int[nnz_symm];
        val_symm = new double[nnz_symm];

        //With NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row) {
            int idx_symm = rowPtr_symm[row];
            for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx) {
                if(col[idx]>=row) {
                    val_symm[idx_symm] = val[idx];
                    col_symm[idx_symm] = col[idx];
                    ++idx_symm;
                }
            }
        }
    }
    return true;
}


void sparsemat::splitMatrixToLDU()
{

    int* L_rowPtr = new int[nrows+1];
    int* U_rowPtr = new int[nrows+1];

    L_rowPtr[0] = 0;
    U_rowPtr[0] = 0;

    //NUMA init
#pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
    {
        L_rowPtr[row+1] = 0;
        U_rowPtr[row+1] = 0;
    }

    int L_nnz = 0;
    int U_nnz = 0;
    int D_nnz = 0;
    for(int row=0; row<nrows; ++row)
    {
        for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
        {
            if(col[idx] > row)
            {
                ++U_nnz;
            }
            else if(col[idx] == row)
            {
                ++D_nnz;
            }
            else
            {
                ++L_nnz;
            }
        }
        L_rowPtr[row+1] = L_nnz;
        U_rowPtr[row+1] = U_nnz;
    }
    if(D_nnz != nrows)
    {
        printf("Error in splitting matrix to LU\n");
        return;
    }

    double* L_val = new double[L_nnz];
    int* L_col = new int[L_nnz];
    double* U_val = new double[U_nnz];
    int* U_col = new int[U_nnz];
    double* D_val = new double[D_nnz];
    //with NUMA init
#pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
    {
        int L_ctr = L_rowPtr[row];
        int U_ctr = U_rowPtr[row];
        for(int idx=rowPtr[row]; idx<rowPtr[row+1]; ++idx)
        {
            if(col[idx]>row)
            {
                U_col[U_ctr] = col[idx];
                U_val[U_ctr] = val[idx];
                ++U_ctr;
            }
            else if(col[idx] == row)
            {
                D_val[row] = val[idx];
            }
            else
            {
                L_col[L_ctr] = col[idx];
                L_val[L_ctr] = val[idx];
                ++L_ctr;
            }
        }
    }

    L = new sparsemat;
    U = new sparsemat;
    L->initCover(nrows, L_nnz, L_val, L_rowPtr, L_col);
    U->initCover(nrows, U_nnz, U_val, U_rowPtr, U_col);
    D = D_val;
}

bool sparsemat::isSymmetric()
{
#ifdef RACE_USE_SPMP
    SpMP::CSR *csr = NULL;
    csr = new SpMP::CSR(nrows, nrows, rowPtr, col, val);
    return csr->isSymmetric(true, true);
#else
    printf("Please link with SpMP library to check for symmetry.\n");
    return -1;
#endif
}

void sparsemat::doRCM()
{
#ifdef RACE_USE_SPMP
    int orig_threads = 1;
    printf("Doing RCM permutation\n");
#pragma omp parallel
    {
        orig_threads = omp_get_num_threads();
    }
    omp_set_num_threads(1);

    SpMP::CSR *csr = NULL;
    csr = new SpMP::CSR(nrows, nrows, rowPtr, col, val);
 //   rcmPerm = new int[nrows];
 //    rcmInvPerm = new int[nrows];
    if(csr->isSymmetric(false,false))
    {
        rcmPerm = new int[nrows];
        rcmInvPerm = new int[nrows];
        csr->getRCMPermutation(rcmInvPerm, rcmPerm);
    }
    else
    {
        printf("Matrix not symmetric RCM cannot be done\n");
    }
    omp_set_num_threads(orig_threads);
    delete csr;
#else
    printf("Please link with SpMP library to enable RCM permutation\n");
#endif
}

void sparsemat::doRCMPermute()
{
    doRCM();
    permute(rcmPerm, rcmInvPerm);
    if(finalPerm)
    {
        delete [] finalPerm;
        delete [] finalInvPerm;
    }
    finalPerm = rcmPerm;
    finalInvPerm = rcmInvPerm;
    rcmPerm = NULL;
    rcmInvPerm = NULL;
}

int sparsemat::prepareForPower(int highestPower, double cacheSize, int nthreads, int smt, PinMethod pinMethod, int globalStartRow, int globalEndRow, std::string mtxType)
{
    //permute(rcmInvPerm, rcmPerm);
    //rcmPerm = NULL;
    //rcmInvPerm = NULL;
    INIT_TIMER(pre_process_kernel);
    START_TIMER(pre_process_kernel);
    ce = new Interface(nrows, nthreads, RACE::POWER, rowPtr, col, symm_hint, smt, pinMethod, rcmPerm, rcmInvPerm);
    //ce->RACEColor(highestPower, cacheSize);
    ce->RACEColor(highestPower, cacheSize*1024*1024, 2, mtxType);
    //ce->RACEColor(highestPower, cacheSize*1024*1024, 2, mtxType, 3);
    if ((globalStartRow != -1) && (globalEndRow != -1))
        ce->passGlobalRows(globalStartRow, globalEndRow);
    STOP_TIMER(pre_process_kernel);
    printf("Pre-processing time: cache size = %f, power = %d, RACE pre-processing time = %fs\n", cacheSize, highestPower, GET_TIMER(pre_process_kernel));

    int *perm, *invPerm, permLen;
    ce->getPerm(&perm, &permLen);
    ce->getInvPerm(&invPerm, &permLen);
    permute(perm, invPerm, true);

    if(finalPerm)
    {
        delete [] finalPerm;
        delete [] finalInvPerm;
    }

    finalPerm = perm;
    finalInvPerm = invPerm;

    checkNumVecAccesses(highestPower);
    //delete [] invPerm;
    //delete [] perm;
    //no idea why need it second time w/o perm. 
    //NUMA init work nicely only if this is done; (only for pwtk, others perf
    //degradation))
    //numaInit(true);
    //writeFile("after_RCM.mtx");
    return 1;
}

int sparsemat::maxStageDepth()
{
    return ce->getMaxStageDepth();
}

void sparsemat::numaInit(bool RACEalloc)
{
    permute(NULL, NULL, RACEalloc);
}

densemat* sparsemat::permute_densemat(densemat *vec)
{
    densemat* newVec = new densemat(nrows, vec->ncols);
    newVec->setVal(0);

    if(nrows != vec->nrows)
    {
        ERROR_PRINT("Permutation of densemat not possible, dimension of matrix and vector do not match");
    }
    for(int row=0; row<nrows; ++row)
    {
        int perm_row = row;
        if(finalPerm != NULL)
        {
            perm_row = finalPerm[row];
        }
        for(int col=0; col<vec->ncols; ++col)
        {
            newVec->val[col*nrows+row] = vec->val[col*nrows+perm_row];
        }
    }
    return newVec;
}

//symmetrically permute
void sparsemat::permute(int *_perm_, int*  _invPerm_, bool RACEalloc)
{
    double* newVal = (double*)malloc(sizeof(double)*block_size*block_size*nnz);
        //new double[block_size*block_size*nnz];
    int* newCol = (int*)malloc(sizeof(int)*nnz);
        //new int[nnz];
    int* newRowPtr = (int*)malloc(sizeof(int)*(nrows+1));
        //new int[nrows+1];

/*
    double *newVal = (double*) malloc(sizeof(double)*nnz);
    int *newCol = (int*) malloc(sizeof(int)*nnz);
    int *newRowPtr = (int*) malloc(sizeof(int)*(nrows+1));
*/

    newRowPtr[0] = 0;

    printf("Initing rowPtr\n");
    if(!RACEalloc)
    {
        //NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            newRowPtr[row+1] = 0;
        }
    }
    else
    {
        ce->numaInitRowPtr(newRowPtr);
    }

    if(_perm_ != NULL)
    {
        //first find newRowPtr; therefore we can do proper NUMA init
        int _perm_Idx=0;
        printf("nrows = %d\n", nrows);
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            int _perm_Row = _perm_[row];
            for(int idx=rowPtr[_perm_Row]; idx<rowPtr[_perm_Row+1]; ++idx)
            {
                ++_perm_Idx;
            }
            newRowPtr[row+1] = _perm_Idx;
        }
    }
    else
    {
        for(int row=0; row<nrows+1; ++row)
        {
            newRowPtr[row] = rowPtr[row];
        }
    }


    printf("Initing mtxVec\n");
    if(RACEalloc)
    {
        ce->numaInitMtxVec(newRowPtr, newCol, newVal, NULL);
    }

    printf("Finished inting\n");
    if(_perm_ != NULL)
    {
        //with NUMA init
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            //row _perm_utation
            int _perm_Row = _perm_[row];
            for(int _perm_Idx=newRowPtr[row],idx=rowPtr[_perm_Row]; _perm_Idx<newRowPtr[row+1]; ++idx,++_perm_Idx)
            {
                //_perm_ute column-wise also
                //newVal[_perm_Idx] = val[idx];
                newCol[_perm_Idx] = _invPerm_[col[idx]];
                for(int b=0; b<block_size*block_size; ++b)
                {
                    newVal[_perm_Idx+b] = val[idx+b];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for(int row=0; row<nrows; ++row)
        {
            for(int idx=newRowPtr[row]; idx<newRowPtr[row+1]; ++idx)
            {
                //newVal[idx] = val[idx];
                newCol[idx] = col[idx];
                for(int b=0; b<block_size*block_size; ++b)
                {
                    newVal[idx+b] = val[idx+b];
                }
            }
        }
    }

    //free old _perm_utations
    //shouldn't delete since it belongs to Python
    //Probably it's GC will clean up
    //delete[] val;
    //delete[] rowPtr;
    //delete[] col;


    val = newVal;
    rowPtr = newRowPtr;
    col = newCol;
}

//make sure all permutation are actuakly done before entring this routine,
//because no initPerm is supported now
//also outPerm and outInvPerm have to be allocated before entering this
//routine
bool sparsemat::doMETIS(int blocksize, int start_row, int end_row, int *initPerm, int *initInvPerm, int *outInvPerm)
{
#if (defined RACE_HAVE_METIS)
    bool success_flag=true;
    int* rptlocal = (int*) malloc(sizeof(int)*(nrows+1));
    int* collocal = (int*) malloc(sizeof(int)*nnz);
    rptlocal[0] = 0;
    int local_nrows = end_row-start_row;
    int local_nnz = 0;
    int local_row=0;

    for(int row=start_row; row<end_row; ++row, ++local_row)
    {
        int permRow = row;
        if(initPerm)
        {
            permRow = initPerm[row];
        }
        for(int idx=rowPtr[permRow]; idx<rowPtr[permRow+1]; ++idx)
        {
            int permCol = col[idx];
            if(initInvPerm)
            {
                permCol = initInvPerm[permCol];
            }
            //only local entries added to col
            if((permCol>=start_row) && (permCol<end_row))
            {
                collocal[local_nnz] = permCol-start_row; //-start_row to convert to local index
                ++local_nnz;
            }
        }
        rptlocal[local_row+1] = local_nnz;
    }

    //partition using METIS
    int ncon = 1;
    int nparts = (int)(local_nrows/(double)blocksize);
    int objval;
    int *part = (int*) malloc(sizeof(int)*local_nrows);

    printf("partitioning graph to %d parts\n", nparts);
    int metis_ret = METIS_PartGraphKway(&local_nrows, &ncon, rptlocal, collocal, NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, part);
    if(metis_ret == METIS_OK)
    {
        printf("successfully partitioned graph to nparts=%d\n", nparts);
    }
    else
    {
        success_flag=false;
        printf("Error in METIS partitioning\n");
    }

    std::vector<std::vector<int>> partRow(nparts);
    for (int i=0;i<local_nrows;i++) {
        partRow[part[i]].push_back(i);
    }
/*    for(int i=start_row; i<end_row; ++i) {
        outPerm[i]=-1;
        outInvPerm[i]=-1;
    }*/
    int ctr=0;
    for (int partIdx=0; partIdx<nparts; partIdx++)
    {
        int partSize = (int)partRow[partIdx].size(); //partPtr[partIdx+1]-partPtr[partIdx];
        for(int rowIdx=0; rowIdx<partSize; ++rowIdx)
        {
            //find rows in parts
            int local_currRow = partRow[partIdx][rowIdx];
            int currRow = local_currRow + start_row;
            int permCurrRow = currRow;
            if(initPerm)
            {
                permCurrRow = initPerm[currRow];
            }

            outInvPerm[permCurrRow] = start_row+ctr;
            ++ctr;
        }
    }
/*
    for (int i=start_row; i<end_row; i++)
    {
        outPerm[outInvPerm[i]] = i;
    }
*/
    return success_flag;
#else
    return false;
#endif
}

//here openMP threads are pinned according to
//RACE pinning, which is necessary for NUMA init
void sparsemat::pinOMP(int nthreads)
{
    omp_set_dynamic(0);    //  Explicitly disable dynamic teams
    int availableThreads = ce->getNumThreads();
    omp_set_num_threads(availableThreads);

#pragma omp parallel
    {
        int pinOrder = omp_get_thread_num();
        ce->pinThread(pinOrder);
    }
}


NUMAmat::NUMAmat(sparsemat *mat_, bool manual, std::vector<int> splitRows_):mat(mat_)
{
    if(manual)
    {
        splitRows = splitRows_;
    }
    else
    {
        splitRows = getRACEPowerSplit();
    }
    NUMAdomains  = splitRows.size()-1;

    nrows = new int[NUMAdomains];
    nnz = new int[NUMAdomains];
    rowPtr = new int*[NUMAdomains];
    col = new int*[NUMAdomains];
    val = new double*[NUMAdomains];
    for(int domain = 0; domain<NUMAdomains; ++domain)
    {
        int startRow = splitRows[domain];
        int endRow = splitRows[domain+1];
        int currNrows = endRow - startRow;
        rowPtr[domain] = new int[currNrows+1];
    }

    if(!manual)
    {
    //    mat->ce->numaInitRowPtr(rowPtr);
    }

#pragma omp parallel
    {
        int totalThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int threadPerNode = totalThreads/NUMAdomains;
        int domain = tid/threadPerNode;
        int localTid = tid%threadPerNode;

        if(localTid == 0)
        {
            int startRow = splitRows[domain];
            int endRow = splitRows[domain+1];
            int currNrows = endRow - startRow;
            //BCSR not yet for NUMAmat
            nrows[domain] = currNrows;
            //rowPtr[domain] = new int[currNrows+1];
            rowPtr[domain][0] = 0;
            int cur_nnz = 0;
            for(int row=0; row<currNrows; ++row)
            {
                //rowPtr[domain][row+1] = mat->rowPtr[row+1+startRow];

                for(int idx=mat->rowPtr[row+startRow]; idx<mat->rowPtr[row+1+startRow]; ++idx)
                {
                    ++cur_nnz;
                }
                rowPtr[domain][row+1] = cur_nnz;
            }
            nnz[domain] = cur_nnz;
        }
    }
    for(int domain = 0; domain<NUMAdomains; ++domain)
    {
        int cur_nnz = nnz[domain];
        col[domain] = new int[cur_nnz];
        val[domain] = new double[cur_nnz];
    }

    if(!manual)
    {
    //    mat->ce->numaInitMtxVec(rowPtr, col, val, 1);
    }

#pragma omp parallel
    {
        int totalThreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int threadPerNode = totalThreads/NUMAdomains;
        int domain = tid/threadPerNode;
        int localTid = tid%threadPerNode;

        if(localTid == 0)
        {
            int startRow = splitRows[domain];
            int endRow = splitRows[domain+1];
            int currNrows = endRow - startRow;

            for(int row=0; row<currNrows; ++row)
            {
                for(int idx=mat->rowPtr[row+startRow], local_idx=rowPtr[domain][row]; idx<mat->rowPtr[row+1+startRow]; ++idx,++local_idx)
                {
                    col[domain][local_idx] = mat->col[idx];
                    val[domain][local_idx] = mat->val[idx];
                }
            }
        }
    }
    /*
    printf("Mat = \n");
    for(int row=0; row<mat->nrows; ++row)
    {
        printf("row = %d \t", row);
        for(int idx=mat->rowPtr[row]; idx<mat->rowPtr[row+1]; ++idx)
        {
            printf("(%d,%f) ", mat->col[idx], mat->val[idx]);
        }
        printf("\n");
    }

    printf("Sub mat = \n");
    for(int domain=0; domain<NUMAdomains; ++domain)
    {
        printf("domain = %d\n", domain);
        for(int row=0; row<nrows[domain]; ++row)
        {
            printf("row = %d \t", row);
            for(int idx=rowPtr[domain][row]; idx<rowPtr[domain][row+1]; ++idx)
            {
                printf("(%d,%f) ", col[domain][idx], val[domain][idx]);
            }
            printf("\n");
        }
    }
*/
}

std::vector<int> NUMAmat::getRACEPowerSplit()
{
    int *split, splitLen;
    mat->ce->getNumaSplitting(&split, &splitLen);
    std::vector<int> split_vec;
    for(int i=0; i<splitLen; ++i)
    {
        split_vec.push_back(split[i]);
    }

    return split_vec;
}

NUMAmat::~NUMAmat()
{
    for(int domain = 0; domain<NUMAdomains; ++domain)
    {
        if(rowPtr[domain])
        {
            delete[] rowPtr[domain];
        }

        if(col[domain])
        {
            delete[] col[domain];
        }

        if(val[domain])
        {
            delete[] val[domain];
        }
    }
    delete[] rowPtr;
    delete[] col;
    delete[] val;
}

inline void MAT_NUM_VEC_ACCESSES(int start, int end, int pow, int numa_domain, void* args)
{
    DECODE_FROM_VOID(args);

    int nrows=mat->nrows;
    for(int row=start; row<end; ++row)
    {
        x->val[row]++;
        if(x->val[row] != pow)
        {
            if(x->val[row] > pow)
            {
                ERROR_PRINT("Oh oh we have duplicate computations, error at pow=%d, for row=%d. Value I got at x is %f, expected %d. Level start =%d, Level end=%d", pow, row, x->val[row], pow, start, end);
            }
            else
            {
                ERROR_PRINT("Oh oh have some missing computations, error at pow=%d, for row=%d. Value I got at x is %f, expected %d. Level start =%d, Level end=%d", pow, row, x->val[row], pow, start, end);
            }
        }
    }
}

inline void MAT_NUM_VEC_ACCESSES_w_subPower(int start, int end, int pow, int subPow, int numa_domain, void* args)
{
    DECODE_FROM_VOID(args);

    int nrows=mat->nrows;
    for(int row=start; row<end; ++row)
    {
        x->val[row]++;
        if(x->val[row] != (pow-1)*3+subPow)
        {
            ERROR_PRINT("Oh oh we have duplicate computations, error at pow=%d, subPow=%d, for row=%d. Value I got at x is %f, expected %d. Level start =%d, Level end=%d", pow, subPow, row, x->val[row], (pow-1)*3+subPow, start, end);
        }
    }
}

void sparsemat::checkNumVecAccesses(int power)
{
    densemat* x = new densemat(this->nrows);
    ENCODE_TO_VOID(this, NULL, x);
    int race_power_id = ce->registerFunction(&MAT_NUM_VEC_ACCESSES, voidArg, power);
    //int race_power_id = ce->registerFunction(&MAT_NUM_VEC_ACCESSES_w_subPower, voidArg, power, 3);
    {
        ce->executeFunction(race_power_id);
    }
    DELETE_ARG();
    delete x;
}


