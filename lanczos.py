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
from kernels import *

def min_eig(A, v0, tol, maxit, verbose=True):
    '''
    lanczos.eig_min: compute the smallest eigenvalue and eigenvector of a symmetric sparse matrix.

    lambda = min_eig(A, v0, tol, maxit, verbose)
    Where A is a symmetric scipy.sparse.csr_matrix or sellcs.sellcs_matrix, v0 is a numpy.array of size A.shape[0],
    tol is the convergence tolerance and maxit the maximum number of iterations.
    '''
    alpha = np.zeros(maxit)
    beta  = np.zeros(maxit+1)
    v = clone(v0)
    v_old = clone(v0)
    init(v_old, 0.0)
    w = clone(v0)

    tmp = np.sqrt(dot(v0,v0))
    # v=v0/||v0||_2
    axpby(1/tmp,v0,0.0,v)
    beta[0]=0
    lmin = 0.0
    for k in range(maxit):
        # w = A*v
        if hasattr(A, 'apply'):
            A.apply(v, w)
        else:
            spmv(A, v, w)
        alpha[k] = dot(v,w)
        # v = Av - alpha[k]*v - beta[k]*v_old
        axpby(-alpha[k], v, 1.0, w)
        axpby(-beta[k],v_old, 1.0, w)
        beta[k+1]=np.sqrt(dot(w,w))
        # v_old = v, v = v/beta[k+1].
        # Note that this is a very simple Lanczos implementation:
        # To get eigenvectors one would have to store the basis V
        # and reorthogonalize from time to time, in particular if
        # multiple eigenpairs are sought.
        v_tmp = v; v = v_old; v_old = v_tmp
        axpby(1.0/beta[k+1],w,0.0,v)
        lmin_old = lmin
        ritz_values, ritz_vectors = scipy.linalg.eigh_tridiagonal(alpha[0:k+1], beta[1:k+1], eigvals_only=False, select='a', select_range=None, check_finite=True, tol=0.0, lapack_driver='auto')
        lmin = np.min(ritz_values)
        ldiff = abs((lmin-lmin_old)/lmin)
        if verbose:
            print('%d\t%e %e'%(k, lmin, ldiff))
        if (ldiff<tol):
            print('Smallest eigenvalue converged to given tolerance.')
            break
    return lmin


if __name__=='__main__':
    from pels_args import get_argparser
    from matrix_generator import create_matrix
    from scipy.sparse import csr_matrix
    from scipy.io import mmread
    from sellcs import *
    import gc

    parser = get_argparser()

    args = parser.parse_args()

    compile_all()

    if args.seed is not None:
        np.random.seed(args.seed)

    ## **Note:** The Python garbage collector (gc)
    ## can kill the performance of the C kernels
    ## for some obscure reason (possibly a conflict
    ## between Numba/LLVM and other compilers like GCC).
    ## For the pure Python/numba/cuda kernels, this is not
    ## the case, but if you are facing obvious performnace
    ## problems with the C kernels, you may want to disalbe
    ## garbage collection:
    gc.disable()

    if args.matfile != 'None':
        if args.matgen!='None':
            print('got both -matfile and -matgen, the latter will be ignored.')
        if not args.matfile.endswith(('.mm','.mtx','.mm.gz','.mtx.gz')):
            raise(ValueError('Expecting MatrixMarket file with extension .mm[.gz] or .mtx[.gz]'))
        A = csr_matrix(mmread(args.matfile))
    elif args.matgen != 'None':
        A = create_matrix(args.matgen)

    N = A.shape[0]
    v0 = np.random.rand(N)

    sigma=1

    if args.fmt=='SELL':
        C=args.C
        sigma=args.sigma
        A = sellcs_matrix(A_csr=A, C=C, sigma=sigma)
        v0 = v0[A.permute]
        print('Matrix format: SELL-%d-%d'%(C,sigma))
    else:
        print('Matrix format: CSR')

    # If a GPU is found, this copies data to the GPU
    # and creates cuda arrays in A.
    # On the CPU, this checks the '-numa' flag and if found,
    # copies the data arrays with first-touch initialization
    # to try to optimize memory accesses.
    A = to_device(A)
    v0 = to_device(v0)

    if available_gpus()>0:
        type = 'gpu'
    else:
        type = 'cpu'

    print('Will run on '+type)

    lmin = min_eig(A,v0,args.tol, args.maxit, verbose=True)
    print('Smallest eigenvalue computed: %e'%(lmin))
    perf_report(type)
