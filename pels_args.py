from argparse import *

def get_argparser():
    '''
    pels_argparser() returns an argparse.ArgumentParser object that
    offers some useful settings for, e.g.,:

    - selecting the kernels to be used
    - selecting the sparse matrix format (CSR or SELL-C-sigma)
    - setting solver parameters like number of iterations and convergence tolerance
    - etc.

    For a full list, run your driver with the --help optino.
    '''
    parser = ArgumentParser(description='Run a CG benchmark.')
    parser.add_argument('-matfile', type=str, default='None',
                    help='MatrixMarket filename for matrix A')
    parser.add_argument('-matgen', type=str, default='None',
                    help='Matrix generator string  for matrix A (e.g., "Laplace128x64"')
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
    parser.add_argument('-c_kernels', action=BooleanOptionalAction,
                    help='Use C/OpenMP kernels for the CPU (instead of Numba).'
                         '-c_kernels is required for using RACE, but may cause a conflict with the GPU (numba) kernels.')

    parser.add_argument('-numa', action=BooleanOptionalAction,
                    help='Do first-touch initialization for Non-Uniform Memory Access (NUMA, e.g. multi-socket CPUs). May cause some additional copy operations during setup.')

    return parser
