import os
from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes, as_array

mkl_functions = None
have_MKL = False

os.system("cd INTEL_MKL && make clean && make")
#mklroot=os.environ.get('MKLROOT')
#so_file = mklroot+"/lib/intel64/libmkl_intel_lp64.so"
#os.environ['LD_LIBRARY_PATH'] = mklroot+"/lib/intel64"
so_file = "INTEL_MKL/lib/libmkl.so"

#define only functions that are necessary for our case
mkl_functions = CDLL(so_file, mode=1)
have_MKL = True
c_double_p = POINTER(c_double)
c_int_p = POINTER(c_int)

mkl_functions.mkl_sparse_trsv_init.restype = c_void_p
mkl_functions.mkl_sparse_trsv_init.argtypes = [c_int, c_int, c_int_p, c_int_p, c_double_p]
mkl_functions.mkl_sparse_trsv_execute.argtypes = [c_int, c_void_p, c_double_p, c_double_p]
mkl_functions.mkl_sparse_trsv_free.argtypes = [c_void_p]

def mkl_ilu0_setup(rptrA, colA, valA, ilu, ipar, dpar, ierr):
    N=rptrA.shape[0]-1
    #convert everything to pointer, because this MKL routine takes only pointers
    N_ptr = np.zeros(1,dtype='int32')
    N_ptr[0] = N
    ierr_ptr = np.zeros(1,dtype='int32')
    ierr_ptr[0] = ierr
    mkl_functions.dcsrilu0(as_ctypes(N_ptr), as_ctypes(valA), as_ctypes(rptrA), as_ctypes(colA), as_ctypes(ilu), as_ctypes(ipar), as_ctypes(dpar), as_ctypes(ierr_ptr))
    ierr = ierr_ptr[0]

def mkl_sparse_trsv_setup(lower, rptrA, colA, valA):
    N=rptrA.shape[0]-1
    handle = mkl_functions.mkl_sparse_trsv_init(lower, N, as_ctypes(rptrA), as_ctypes(colA), as_ctypes(valA))
    return handle
    
def mkl_sparse_trsv(lower, handle, b, x):
    mkl_functions.mkl_sparse_trsv_execute(lower, handle, as_ctypes(b), as_ctypes(x))
    
def mkl_sparse_trsv_free(handle):
    mkl_functions.mkl_sparse_trsv_free(handle)