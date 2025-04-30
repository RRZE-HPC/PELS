#include <mkl.h>
#include <stdio.h>

//Solves for U*x=b (is_lower=0) or L*x=b (is_lower=1)
void* mkl_sparse_trsv_init(int is_lower, int nrows, int* rowPtr, int* col, double* val)
{
    sparse_matrix_t* A = (sparse_matrix_t*) mkl_malloc(sizeof(sparse_matrix_t), 128);
    mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, nrows, nrows, rowPtr, rowPtr+1, col, val);
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = (is_lower==1)?SPARSE_FILL_MODE_LOWER:SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    int operation = SPARSE_OPERATION_NON_TRANSPOSE;
    mkl_sparse_order(*A);
    //setting 20 iterations as the expected iteration count
    mkl_sparse_set_sv_hint (*A, operation, descr, 20);
    mkl_sparse_optimize(*A);
    void* handle = (void*) A;
    return handle;
}

//Solves for U*x=b (is_lower=0) or L*x=b (is_lower=1)
void mkl_sparse_trsv_execute(int is_lower, void* handle, double *b, double *x)
{
    sparse_matrix_t* A = (sparse_matrix_t*)handle;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = (is_lower==1)?SPARSE_FILL_MODE_LOWER:SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    int operation = SPARSE_OPERATION_NON_TRANSPOSE;
    mkl_sparse_d_trsv (operation, 1.0, *A, descr, b, x);
}

void mkl_sparse_trsv_free(void* handle)
{
    sparse_matrix_t* A = (sparse_matrix_t*)handle;
    mkl_free(A);
}