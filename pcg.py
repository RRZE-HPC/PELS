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
from kernels import *

def cg_solve(A, M, b, x0, tol, maxit, verbose=True, x_ex=None):
    '''
    Right-Preconditioned CG Solver
    A: System matrix
    M: Preconditioner (must have an .apply(in, out) or similar method)
    '''
    x = clone(b)
    r = clone(b)
    p = clone(b)
    q = clone(b)

    if M is None:
        z = r
    else:
        z = clone(b) # Preconditioned residual

    if x_ex is not None:
        print('PerfWarning: Extra operations enabled for error norm.')
        err = clone(b)

    tol2 = tol*tol

    # Initial Residual: r = b - A*x0
    axpby(1.0, x0, 0.0, x)
    if hasattr(A, 'apply'):
        A.apply(x, r)
    else:
        spmv(A, x, r)
    axpby(1.0, b, -1.0, r)

    # Initial Preconditioning: z = M^-1 * r
    if M is not None:
        M.apply(r, z)

    # Initial search direction: p = z
    axpby(1.0, z, 0.0, p)

    # rho = <r, z> (The preconditioned scalar)
    rho = dot(r, z)

    # We use the true residual norm for the stopping criterion
    if M is not None:
        res_norm_sq = dot(r, r)
    else:
        res_norm_sq = rho

    if verbose:
        print('%d\t%e'%(0, np.sqrt(res_norm_sq)))

    for iter in range(maxit + 1):

        # Check stopping criteria on the true residual
        if res_norm_sq < tol2:
            break

        # q = A * p
        if hasattr(A, 'apply'):
            A.apply(p, q)
        else:
            spmv(A, p, q)

        # alpha = <r, z> / <p, Ap>
        pq = dot(p, q)
        alpha = rho / pq

        # x = x + alpha * p
        axpby(alpha, p, 1.0, x)

        # r = r - alpha * q
        axpby(-alpha, q, 1.0, r)

        # Update preconditioned residual: z = M^-1 * r
        if M is not None:
            M.apply(r, z)

        rho_old = rho
        rho = dot(r, z)
        if M is not None:
            res_norm_sq = dot(r, r)
        else:
            res_norm_sq = rho

        if verbose:
            if x_ex is not None:
                axpby(1.0, x, 0.0, err)
                axpby(-1.0, x_ex, 1.0, err)
                err_norm = np.sqrt(dot(err, err))
                print('%d\t%e\t%e'%(iter+1, np.sqrt(res_norm_sq), err_norm))
            else:
                print('%d\t%e'%(iter+1, np.sqrt(rho)))

        # beta = <r_new, z_new> / <r_old, z_old>
        beta = rho / rho_old

        # p = z + beta * p
        axpby(1.0, z, beta, p)

    res_norm_sq = dot(r, r)
    return x, np.sqrt(res_norm_sq), iter

import numba
from numpy.linalg import norm
from sellcs import sellcs_matrix
import cusolver_ichol

import precon

from matrix_generator import create_matrix
from pels_args import *

def pcg_demo(args_dict={}):
    '''
    This is a driver routine that can be called from a Jupyter notebook,
    with argujments in a Python dict. For example:

    d = {'matrix': 'Laplace100x100', 'fmt': 'SELL', 'C': 128, 'maxit': 500, 'tol': 1.0e-3}
    pcg_demo(d)

    Additional arguments are read from the command-line, and defaults are set for any unspecified parameters.
    The '--help' option can be used to get information about all possible parameters.
    '''
    args = get_pcg_args(args_dict)

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.matrix is not None:
        if type(args.matrix) == scipy.sparse.csr_matrix:
            A = args.matrix
        else:
            A = create_matrix(args.matrix)
    else:
        raise Exception('You must specify the -matrix option. Use --help for more information.')
    N = A.shape[0]

    if args.solfile!='None':
        x_ex=mmread(args.solfile).reshape(N)
    else:
        x_ex=np.random.rand(N)

    if args.rhsfile!='None':
        b=mmread(args.rhsfile).reshape(N)
    else:
        b=A*x_ex

    x0 = np.zeros(N,dtype='float64')

    print('norm of rhs: %e'%(norm(b)))
    print('rel. residual of given solution: %e'%(norm(A*x_ex-b)/norm(b)))

    tol = args.tol
    maxit = args.maxit

    sigma=1

    A_csr = A # we may need it for creating the preconditioner
              # in case the user wants a SELL-C-sigma matrix.

    if args.fmt=='SELL':
        C=args.C
        sigma=args.sigma
        A = sellcs_matrix(A_csr=A_csr, C=C, sigma=sigma)
        b = b[A.permute]
        print('Matrix format: SELL-%d-%d'%(C,sigma))
        A_csr = A_csr[A.permute[:,None], A.permute]
    else:
        print('Matrix format: CSR')

    M = None

    x0 = to_device(x0)
    x_ex = to_device(x_ex)
    b  = to_device(b)
    A  = to_device(A)

    # take compilation time out of the balance:
    compile_all()

    # we want to make sure what we measure during CG in total
    # is consistent with the sum of the kernel calls and their
    # runtime as predicted by the roofline model, so reset all
    # counters and timers:
    reset_counters()

    M=None
    t0 = perf_counter()

    if args.precon is not None:
        # setup preconditioner...
        if   args.precon == 'Jacobi' or args.precon == 'jacobi':
            M = precon.Jacobi(A_csr)
        elif args.precon == 'SGS':
            M = precon.SymmetricGaussSeidel(A_csr, fast_trsv=args.fast_trsv)
        elif args.precon.startswith('IC'):
            if args.precon.endswith('p'):
                # First apply a fill-reducing ordering to the linear system
                # -- otherwise the performance is horrible.
                A_csr, x_ex, b = precon.apply_metis_preordering(A_csr, from_device(x_ex), from_device(b))
                A_csr    = to_device(A_csr)
                A = A_csr
                x_ex = to_device(x_ex)
                b    = to_device(b)
            if args.precon.startswith('IC0'):
                M = cusolver_ichol.IChol(A_csr, args.ic_poly)
            else:
                M = precon.IChol(A_csr, args.ic_fill, args.ic_droptol, poly_k=args.ic_poly, fast_trsv=args.fast_trsv)
        else:
            raise Exception("Unsupported parameter: -precon='"+args.precon+"'")
        if args.fmt == 'SELL' and A.sigma!=1:
            raise Exception("Preconditioning not implemented for SELL-C-simga format with sigma>1")

    x_ex_in = None
    if args.printerr:
        x_ex_in = x_ex

    t0_soln = perf_counter()
    x, relres, iter = cg_solve(A,M, b,x0,tol,maxit, x_ex=x_ex_in)
    t1_soln = perf_counter()

    t1 = perf_counter()
    t_CG = t1-t0

    if M is not None:
        t_soln = t1_soln-t0_soln

    x = to_host(x)

    print('number of CG iterations: %d'%(iter))
    res = np.empty_like(x)
    res = b - A_csr@x
    print('relative residual of computed solution: %e'%(norm(res)/norm(b)))

    if args.fmt=='SELL' and sigma>1:
        x = x[A.unpermute]

    print('relative error of computed solution: %e'%(norm(x-x_ex)/norm(x_ex)))

    print()
    perf_report()

    if M is not None:
        t_spmv = time['spmv']/calls['spmv']
        t_setup = precon.time['setup']
        t_apply = precon.time['apply']
        t_apply_per_call = t_apply / precon.calls['apply']
        spmvs_per_apply = t_apply_per_call/t_spmv
        print('Total time for constructing precon: %g seconds (%d spmvs).'%(t_setup, t_setup/t_spmv))
        print('Total time for applying precon: %g seconds (%g spmvs/call).'%(t_apply, spmvs_per_apply))
        print('Total time for solving: %g seconds.'%(t_soln))

    print('Total time for CG: %g seconds.'%(t_CG))

if __name__ == '__main__':
    # by default, the driver gets all arguments from the command-line
    pcg_demo()
