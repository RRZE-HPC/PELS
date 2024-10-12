import os
from ctypes import *
from numpy.ctypeslib import as_ctypes, as_array

mpk_c_functions = None
have_RACE = False

os.system("cd RACE && make -j")
try:
    # import the C library -> c_functions object
    so_file = "RACE/lib/libmpk.so"
    mpk_c_functions = CDLL(so_file)
    have_RACE = True
    c_double_p = POINTER(c_double)
    c_int_p = POINTER(c_int)
    mpk_c_functions.mpk_setup.restype = c_void_p
    mpk_c_functions.mpk_setup.argtypes = [c_int, c_int_p, c_int_p, c_double_p, c_int, c_double, c_int]
    mpk_c_functions.mpk_getPerm.restype = c_int_p
    mpk_c_functions.mpk_getPerm.argtypes = [c_void_p]
    mpk_c_functions.mpk.argtypes = [c_void_p, c_int, c_double_p, c_double_p]
    mpk_c_functions.mpk_neumann_apply.argtypes = [c_void_p, c_int, c_double_p, c_double_p]
    mpk_c_functions.mpk_free.argtypes = [c_void_p]
except:
    print('Warning: RACE library could not be built or found, use_RACE will not be supported.')
    have_RACE = False
    pass

def csr_mpk_setup(rptrA, colA, valA, power, cacheSize, split):
    N=rptrA.shape[0]-1
    p=mpk_c_functions.mpk_setup(N, as_ctypes(rptrA), as_ctypes(colA), as_ctypes(valA), power, cacheSize, split)
    return p

def csr_mpk_get_perm(voidA, N):
    return as_array(mpk_c_functions.mpk_getPerm(voidA), shape=(N,))

def csr_mpk(voidA, power, x, y):
    mpk_c_functions.mpk(voidA, power, as_ctypes(x), as_ctypes(y))

def csr_mpk_neumann_apply(voidA, k, x, y):
    mpk_c_functions.mpk_neumann_apply(voidA, k, as_ctypes(x), as_ctypes(y))

def csr_mpk_free(voidA):
    mpk_c_functions.mpk_free(voidA)
