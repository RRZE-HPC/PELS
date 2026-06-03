from argparse import *

def get_argparser():
    '''
    pels_argparser() returns an argparse.ArgumentParser object that
    offers some useful settings for, e.g.,:

    - selecting the sparse matrix format (CSR or SELL-C-sigma)
    - setting solver parameters like number of iterations and convergence tolerance
    - etc.

    For a full list, run your driver with the --help optino.
    '''
    parser = ArgumentParser(description='Run a CG benchmark.',
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('-matfile', type=str, default='None',
                    help='MatrixMarket filename for matrix A')
    parser.add_argument('-matgen', type=str, default='None',
                    help='Matrix generator string  for matrix A. E.g., "Laplace128x64", '+
                         '"Laplace50x50x50", or "LinElast100x50" (latter requires pyamg)')
    parser.add_argument('-maxit', type=int, default=1000,
                    help='Maximum number of CG iterations allowed.')
    parser.add_argument('-tol', type=float, default=1e-6,
                    help='Convergence criterion: ||b-A*x||_2/||b||_2<tol')
    parser.add_argument('-fmt', type=str, default='CSR',
                    help='Sparse matrix format to be used [CSR, SELL]')
    parser.add_argument('-C', type=int, default=1,
                    help='Chunk size C for SELL-C-sigma format.')
    parser.add_argument('-sigma', type=int, default=1,
                    help='Sorting scope sigma for SELL-C-sigma format.')
    parser.add_argument('-seed', type=int, default=None,
                    help='Random seed to make runs reproducible')
    parser.add_argument('-precon', type=str, default=None,
                    help='Preconditioner to be used [None,Jacobi,SGS,IC0,ILU0]\n'+
                         'Jacobi is simple diagonal scaling,\n'+
                         'SGS is Symmetric Gauss-Seidel (which involves triangular solves)\n'+
                         'IC is an incomplete Cholesky factorization (computed on the CPU).\n')
    parser.add_argument('-ic_fill', type=int, default=1,
                    help='With -precon=IC or ILU, set the level-of-fill.')
    parser.add_argument('-ic_droptol', type=float, default=0.0,
                    help='With -precon=IC or ILU, set the relative drop tolerance.')
    parser.add_argument('-ic_poly', type=int, default=-1,
                   help='combine -ilu_poly=<k> with -precon=IC to replace the forward/backward triangular solves\n'+
                   'by a degree-k Neumann polynomial (k spmvs with L and L^T per CG iteration)')

    # add driver-specific command-line arguments for polynomial preconditioning with or without RACE:
    parser.add_argument('-printerr', action=BooleanOptionalAction,
                    help='Besides the residual norm, also compute and print the error norm.')
    parser.add_argument('-rhsfile', type=str, default='None',
                    help='MatrixMarket filename for right-hand side vector b')
    parser.add_argument('-solfile', type=str, default='None',
                    help='MatrixMarket filename for exact solution x')

    return parser

# Given an ArgumentParser P, parses the command-line and afterwards
# overwrites those entries found in the dictionary A. Returns a complete

def get_arg_dict(P, A={}):

    args = P.parse_args()
    args_dict = vars(args)

    A_default = vars(P)
    invalid_keys = set(B) - set(A)

    if invalid_keys:
        raise ValueError(f"Unexpected arguments encountered: {invalid_keys}")

    # If valid, merge them (Python 3.9+)
    A = A | A_default
    return A
