#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

from time import perf_counter
import numpy as np
import scipy
import numba

import sellcs

import sys
import os

have_c_kernels = False
have_RACE = False

if '-c_kernels' in sys.argv or '-use_RACE' in sys.argv or 'USE_C_KERNELS' in os.environ or 'USE_RACE' in os.environ:
    try:
        import kernels_c as cpu
        have_c_kernels=True
        print('Using C kernels on CPU')
    except:
        import kernels_cpu as cpu
        print('Failed to import/compile C kernels, you may need to adjust "make.inc".\n'+
              'Falling back to numba-compiled kernels')
else:
    print('Using Numba kernels on CPU')
    import kernels_cpu as cpu

if '-use_RACE' in sys.argv:
    if not have_c_kernels:
        print('-use_RACE is ignored because C kernels are not available.')
    else:
        import race_mpk
        have_RACE = race_mpk.have_RACE

# for benchmarking numpy/scipy implementations,
# uncomment this line instead of the above:
#import kernels_numpy as cpu

try:
    from numba import cuda
    from numba.cuda import is_cuda_array
    import kernels_gpu as gpu
except:
    print('Could not load cuda module and/or kernels')
    gpu = cpu
    cuda = None

def available_gpus():
    if cuda is None:
        return 0
    if cuda.is_available()==False:
        return 0
    return len(cuda.gpus)

def compile_all():
    n=10
    x=np.ones(n,dtype='float64')
    y=np.ones(n,dtype='float64')
    a=numba.float64(1.0)
    b=numba.float64(1.0)
    A1=scipy.sparse.csr_matrix(scipy.sparse.rand(n,n,0.6))
    A2=sellcs.sellcs_matrix(A1, C=1, sigma=1)

    # compile CPU kernels:
    init(x,a)
    z = clone(x)
    s=dot(x,y)
    axpby(a,x,b,y)
    spmv(A1,x,y)
    spmv(A2,x,y)
    # compile GPU kernels:
    if available_gpus()>0:
        x = to_device(x)
        tmp = from_device(x)
        y = to_device(x)
        A1 = to_device(A1)
        tmp= from_device(A1)
        A2 = to_device(A2)
        tmp = from_device(A2)
        init(x,a)
        z = clone(x)
        s=dot(x,y)
        axpby(a,x,b,y)
        spmv(A1,x,y)
        spmv(A2,x,y)
        diag_spmv(A1,x,y)
    reset_counters()

def memory_benchmarks(type):
    if type=='cpu':
        return cpu.memory_benchmarks()
    elif type=='gpu':
        return gpu.memory_benchmarks()
    else:
        raise('type should be "cpu" or "gpu"')

# total number of calls
calls = {'spmv': 0, 'axpby': 0, 'dot': 0, 'init': 0}
# total elapsed time in seconds
time = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total loaded data in GB
load = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total stored data in GB
store = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}
# total floating point operations [GFlop]
flop = {'spmv': 0.0, 'axpby': 0.0, 'dot': 0.0, 'init':0.0}

# which benchmark to use for predicting memory bandwidth achievable by an operation.
# Benchmark values are currently hard-coded into kernels_cpu.py and kernels_gpu.py for Sapphire Rapids and A100, resp.
bench_map = {'spmv': 'triad', 'axpby': 'triad', 'dot': 'load', 'init': 'store'}

def reset_counters():
    for k in calls.keys():
        calls[k] = 0.0
        time[k] = 0.0
        load[k] = 0.0
        store[k] = 0.0
        flop[k] = 0.0

def same_array(x,y):
    '''
    returns 1 if the C pointer of the two arrays is identical, 0 otherwise
    '''
    if hasattr(x,'__cuda_array_interface__'):
        return int(x.__cuda_array_interface__['data'][0]==y.__cuda_array_interface__['data'][0])
    elif hasattr(x,'__array_interface__'):
        return int(x.__array_interface__['data'][0]==y.__array_interface__['data'][0])
    else:
        return False

def to_device(A):
    '''
    If a GPU is found, this creates CUDA arrays and copies data to the device.
    On the CPU, we check if '-numa' is set on the command-line, and if so,
    copy the data arrays with correct first-touch initialization.
    '''
    if available_gpus()>0:
        return gpu.to_device(A)
    else:
        if '-numa' in sys.argv:
            A = copy(A)
        return A

def from_device(A):
    if available_gpus()>0:
        return gpu.from_device(A)
    else:
        return A

def to_host(A):
    if cuda and is_cuda_array(A):
        return A.copy_to_host()
    elif type(A)==scipy.sparse.csr_matrix or type(A)==sellcs.sellcs_matrix:
        if available_gpus()>0:
            A.indptr = A.cu_indptr.copy_to_host()
            A.data = A.cu_data.copy_to_host()
            A.indices = A.cu_indices.copy_to_host()
    return A

def spmv(A, x, y):
    t0 = perf_counter()
    if cuda and is_cuda_array(x):
        if not hasattr(A, 'cu_data'):
            print('PerfWarning: copying matrix data to device in spmv call. Manually call kernels.to_device(A) to avoid this.')
            A = to_device(A)
        run_on = gpu
        data = A.cu_data
        indptr = A.cu_indptr
        indices = A.cu_indices
    else:
        run_on = cpu
        data = A.data
        indptr = A.indptr
        indices = A.indices
    if type(A)==scipy.sparse.csr_matrix:
        run_on.csr_spmv(data, indptr, indices, x, y)
    elif type(A)==sellcs.sellcs_matrix:
        run_on.sell_spmv(data, indptr, indices, A.C, x, y)
    else:
        raise TypeError('spmv wrapper only implemented for scipy.sparse.csr_matrix or sellcs.sellcs_matrix')
    t1 = perf_counter()
    time['spmv']  += t1-t0
    calls['spmv'] += 1
    load['spmv']  += 12*A.nnz+8*(A.shape[0]+A.shape[1])
    store['spmv'] += 8*A.shape[0]
    flop['spmv'] += 2*A.nnz

def spmv_c(A, x, y):
    '''
    This function explicitly calls the C variant of the CSR spmv.
    It is meant only for testing and benchmarking, if you want to use
    the C kernels, import kernels_c as cpu at the beginning of this module.
    '''
    t0 = perf_counter()
    data = A.data
    indptr = A.indptr
    indices = A.indices
    if type(A)==scipy.sparse.csr_matrix:
        csr_spmv_c(A,x,y)
    else:
        raise TypeError('spmv C wrapper only implemented for scipy.sparse.csr_matrix')
    t1 = perf_counter()
    time['spmv']  += t1-t0
    calls['spmv'] += 1
    if calls['spmv']>0:
        load['spmv']  += 12*A.nnz+8*(A.shape[0]+A.shape[1])
        store['spmv'] += 8*A.shape[0]
        flop['spmv'] += 2*A.nnz

def diag_spmv(A, x, y):
    if cuda and is_cuda_array(x):
        gpu.vscale(A.cu_data, x, y)
    else:
        cpu.vscale(A.data.reshape(x.size), x, y)

def mpk_get_perm(mpk_handle, N):

    if not have_RACE:
        raise AssertionError('RACE is not available, you may need to add the -use_RACE flag and/or install the RACE library.')
    return race_mpk.csr_mpk_get_perm(mpk_handle, N)

def mpk_setup(A, power, cacheSize, split):
    if not have_RACE:
        raise AssertionError('RACE is not available, you may need to add the -use_RACE flag and/or install the RACE library.')
    if type(A)==scipy.sparse.csr_matrix:
        data = A.data
        indptr = A.indptr
        indices = A.indices
        mpk_handle=race_mpk.csr_mpk_setup(indptr, indices, data, power, cacheSize, split)
        perm = mpk_get_perm(mpk_handle, A.shape[0])
        A = A[:,perm][perm,:]
        return mpk_handle, A

def mpk_free(mpk_handle):
    if not have_RACE:
        raise AssertionError('RACE is not available, you may need to add the -use_RACE flag and/or install the RACE library.')
    race_mpk.csr_mpk_free(mpk_handle)

def mpk_neumann_apply(polyHandle, x, y):
    t0 = perf_counter()
    k= polyHandle.k
    race_mpk.csr_mpk_neumann_apply(polyHandle.mpkHandle, k, x, y)
    t1 = perf_counter()
    time['spmv']  += t1-t0
    calls['spmv'] += 2*k+1
    if calls['spmv']>0:
        load['spmv']  += (k+1)*(12*polyHandle.A1.nnz)-2*k*8*(polyHandle.A1.shape[1])+(2*k+1)*8*(polyHandle.A1.shape[0]+polyHandle.A1.shape[1])
        store['spmv'] += (2*k+1)*8*polyHandle.A1.shape[0]
        flop['spmv'] += (k+1)*2*polyHandle.A1.nnz-(2*k*2*polyHandle.A1.shape[1])


def clone(v):
    w = None
    if cuda and is_cuda_array(v):
        w = cuda.device_array(shape=v.shape,dtype=v.dtype)
    else:
        w = np.empty_like(v)
        # first-touch initialization
        cpu.init(w,0.0)
    return w

def copy(X):
    '''
    Copy a vector or matrix (csr_matrix or sellcs_matrix)
    that may live on a GPU, and assure first-touch initialization
    on the CPU.
    '''
    if cuda and is_cuda_array(X):
        Y = cuda.device_array_like(X)
        Y[:] = X[:]
        return Y
    elif type(X) == np.ndarray:
        return cpu.copy_vector(X)
    elif type(X) == scipy.sparse.csr_matrix or type(X) == sellcs.sellcs_matrix:
        data, indices, indptr = cpu.copy_csr_arrays(X.data, X.indptr, X.indices)
        if type(X) == scipy.sparse.csr_matrix:
            A = scipy.sparse.csr_matrix((data, indices, indptr), shape=X.shape)
        elif type(X) == sellcs.sellcs_matrix:
            permute = cpu.copy_vector(X.permute)
            unpermute = cpu.copy_vector(X.unpermute)
            A = sellcs.sellcs_matrix(A_arrays=(data, indices, indptr, permute, unpermute,X.nnz), shape=X.shape, C=X.C, sigma=X.sigma)
        if hasattr(X, 'cu_data'):
            A.cu_data = X.cu_data.copy()
        if hasattr(X, 'cu_indices'):
            A.cu_indices = X.cu_indices.copy()
        if hasattr(X, 'cu_indptr'):
            A.cu_indptr = X.cu_indptr.copy()
        return A

def init(v, val):
    t0 = perf_counter()
    if cuda and is_cuda_array(v):
        gpu.init(v,val)
    else:
        cpu.init(v,val)
    t1 = perf_counter()
    calls['init'] += 1
    time['init']  += t1-t0
    store['init'] += 8*v.size

def axpby(a,x,b,y):
    t0 = perf_counter()
    if cuda and is_cuda_array(y):
        gpu.axpby(a,x,b,y)
    else:
        cpu.axpby(a,x,b,y)
    t1 = perf_counter()
    time['axpby']  += t1-t0
    calls['axpby'] += 1
    load['axpby']  += (2-same_array(x,y))*8*x.size
    store['axpby'] += 8*x.size
    flop['axpby'] += 2*x.size

def dot(x,y):
    t0 = perf_counter()
    if cuda and is_cuda_array(y):
        s = gpu.dot(x,y)
    else:
        s = cpu.dot(x,y)
    t1 = perf_counter()
    time['dot']  += t1-t0
    calls['dot'] += 1
    load['dot']  += (2-same_array(x,y))*8*x.size
    flop['dot'] += 2*x.size
    return s

def perf_report(type):
    '''
    After running a solver, print a performance summary of the
    kernels in this module (dot, axpby, spmv...). The argument 'type'
    should be either 'cpu' or 'gpu', dependning on which hardware you
    ran. It is used to get some benchmark values from files cpujson or
    gpu.json. You should either adjust these files to match your system,
    or remove them to skip printing the roofline upper bounds.
    '''
    bench = memory_benchmarks(type)
    print('Hardware assumed for Roofline Model: %s'%(bench['label']))
    have_bench = True
    if bench['label'] == 'undefined':
        have_bench = False
        print('(roofline values will be skipped -- use cpu.json and/or gpu.json to provide memory bandwidth data)')
    else:
        print('(note that the hardware info is taken from [cpu|gpu].json, if does not match your system,\n'+
              'you may want to update those files or delete them to skip the roofline prediction)')
    if type == 'cpu':
        nthreads = numba.get_num_threads()
        print('Number of threads: %d'%(nthreads))

    # total measured time
    t_tot  = 0
    # model prediction
    t_mod  = 0
    # total number of functions called
    total_calls = 0

    print('--------\t-----\t---------------\t---------------\t---------------\t---------------')
    print('kernel  \tcalls\t bw_meas       \t bw_roofline   \t t_meas/call   \tt_roofline/call')
    print('========\t=====\t===============\t===============\t===============\t===============')
    for kern in ('dot', 'axpby', 'spmv'):
        if calls[kern]>0:
            t_tot += time[kern]
            total_calls += calls[kern]
            if have_bench:
                print('%8s\t%5d\t%8.4g GB/s\t%8.4g GB/s\t%8.4g s \t%8.4g s '%
                    (kern, calls[kern], (load[kern]+store[kern])*1e-9/time[kern], bench[bench_map[kern]], time[kern]/calls[kern], (load[kern]+store[kern])*1e-9/bench[bench_map[kern]]/calls[kern]))
                t_mod += (load[kern]+store[kern])*1e-9/bench[bench_map[kern]]
            else:
                print('%8s\t%5d\t%8.4g GB/s\t   -    \t%8.4g s \t   -    '%
                    (kern, calls[kern], (load[kern]+store[kern])*1e-9/time[kern], time[kern]/calls[kern]))

    print('--------\t-----\t---------------\t---------------\t---------------\t---------------')
    if have_bench:
        print('%8s\t     \t               \t               \t %8.4g s \t %8.4g s'%('Total',t_tot, t_mod))
    else:
        print('%8s\t     \t               \t               \t %8.4g s \t%8s'%('Total',t_tot, '   n/a  '))
    print('--------\t-----\t---------------\t---------------\t---------------\t---------------')
