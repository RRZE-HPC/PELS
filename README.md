
# PELS: Performance Engineering for (sparse) Linear Solvers Demo

This repository is accompanying material to a [tutorial given at ISC'24](https://blogs.fau.de/hager/tutorials/isc24)
(follow the link to find additional material).
It contains a reasonably simple Python implementation of the SELL-C-sigma sparse matrix format and some simple iterative
methods that heavily rely on Sparse Matrix-Vector Multiplication (SpMVM). The goal is to show that the performance of 
such methods on Multi-Core CPUs and GPUs can be modelled quite accurately using the Roofline performance model.

## Computational backends and performance reporting

- There are both CPU and GPU implementations, using ``numba`` and ``numba.cuda``, respectively.
  The drivers will detect if there is a GPU available and switch to using it automatically.
- Use the ``-c_kernels`` flag to the driver routine to use C/OpenMP implementations of
  basic operations. The rules to comple the kernels are defined in ``Makefile``. By default,
  GCC is used, if you want to use another compiler, adapt it accordingly.


# Setup

The Python code in this repository requires only standard libraries like numpy, numba and scipy.
To make sure they are available, you can use the following command:

```bash
pip install --user -r requirements.txt
```

## Optional requirements

In order to use an NVidia GPU, you also need cuda.
For using the C backend (in particular in combination with RACE, see below), you need a C compiler
like GCC or Intel. We recommend the Intel LLVM compilers (icx and icpx). Edit ``make.inc`` to match your system.
Required C code is automatically compiled when running with ``-use_RACE`` or ``-c_kernels``.
The [RACE library](https://github.com/RRZE-HPC/RACE) must be separately installed.

In order to get a Roofline estimate at the end of a run, you should adapt the files 
``cpu.json`` and/or ``gpu.json`` to your platform. For example, to get the benchmark values
for a single NUMA domain using ``likwid``, you could run:

```bash
$ likwid-bench -t load_avx -W M0:8GB|grep "MByte/s"
$ likwid-bench -t store_avx -W M0:8GB|grep "MByte/s"
$ likwid-bench -t copy_avx -W M0:8GB|grep "MByte/s"
$ likwid-bench -t triad_avx -W M0:8GB|grep "MByte/s"
```

A similar benchmark tool for NVidia GPUs can be found [here](https://github.com/bcumming/cuda-stream).
You can run it with

```bash
./stream -s -n 1000000000
```

If you do not have the correct data for your system, you can delete the two files to skip printing the (incorrect) Roofline prediction.

## Testing the installation

The command ``pytest`` will execute a few simple tests. If a GPU and cuda are available, they will be run on the GPU.
In that case, please ignore warnings about under-utilization of the device.
To specifically test the C backend, use ``env USE_C_KERNELS=1 pytest``. To see all output from the tests (even those that pass), you can use the '-s' flag.

# Example usage

There are two drivers: ``lanczos.py`` contains a very simple Lanczos eigenvalue solver to compute the smallest eigenvalue of a matrix. ``'pcg.py`` can be used to solve a (symmetric and positive definite) linear system by the Conjugate Gradient method, optionally with a simple polynomial preconditioner.
Both drivers can be run with the ``--help`` option to get a list of supported parameters.

## Example 1: Sparse Matrix Formats on the GPU

On a node with an Nvidia A100 GPU, we compute the smallest eigenvalue (ground state energy) of a spin-chain matrix:

```bash
$ python3 lanczos.py -matfile spinSZ26.mm.gz
```

Which may result in output like this:

```bash
Smallest eigenvalue computed: -4.621451e+01
Hardware: NVIDIA A100 80GB PCIe
--------        -----   --------------- --------------- --------------- ---------------
kernel          calls    bw_meas         bw_roofline     t_meas/call    t_roofline/call
========        =====   =============== =============== =============== ===============
     dot          119      734.8 GB/s       1560 GB/s   0.0002265 s     0.0001067 s 
   axpby          178      986.8 GB/s       1690 GB/s   0.0002529 s     0.0001477 s 
    spmv           59      347.4 GB/s       1690 GB/s   0.005936 s       0.00122 s 
--------        -----   --------------- --------------- --------------- ---------------
   Total                                                   0.4222 s         0.111 s
--------        -----   --------------- --------------- --------------- ---------------

```

This matrix is irregular with about 10M rows and 1-27 nonzeros per row.
A smaller variant, SpinSZ22 is shown below:

SpinSZ22 sparsity pattern | SpinSZ22 compressed view                 
--------------------------|---------------------------------------------
<img src="spinSZ22.png" alt="SpinSZ22 pattern" width="400"/> | <img src="spinSZ22_ELL.png" alt="SpinSZ22 compressed form" width="400"/> |

Better SpMV performance on the GPU can be achieved by using the SELL-C-sigma format:

```bash
$ python3 lanczos.py -matfile spinSZ26.mm.gz -fmt SELL -C 128 -sigma 1024
[...]
Smallest eigenvalue computed: -4.621451e+01
Hardware: NVIDIA A100 80GB PCIe
--------        -----   --------------- --------------- --------------- ---------------
kernel          calls    bw_meas         bw_roofline     t_meas/call    t_roofline/call
========        =====   =============== =============== =============== ===============
     dot          109      808.8 GB/s       1560 GB/s   0.0002057 s     0.0001067 s 
   axpby          163      983.8 GB/s       1690 GB/s   0.0002537 s     0.0001477 s 
    spmv           54       1260 GB/s       1690 GB/s   0.001636 s       0.00122 s 
--------        -----   --------------- --------------- --------------- ---------------
   Total                                                   0.1521 s        0.1016 s
--------        -----   --------------- --------------- --------------- ---------------
```

For the larger and very regular 5-point Laplace matrix with 100M rows and columns, the CSR format works almost as well as SELL-C-sigma on this GPU,
and kernel launch latency has less impact, so that the performance of all kernels is quite good:
```bash
$ python lanczos.py -matgen Laplace10000x10000 -tol 1e-3 -fmt CSR
[...]
Smallest eigenvalue computed: 6.847751e-07
Hardware assumed for Roofline Model: NVIDIA A100 80GB PCIe
(note that the hardware info is taken from [cpu|gpu].json, if does not match your system,
you may want to update those files or delete them to skip the roofline prediction)
--------	-----	---------------	---------------	---------------	---------------
kernel  	calls	 bw_meas       	 bw_roofline   	 t_meas/call   	t_roofline/call
========	=====	===============	===============	===============	===============
     dot	 1831	    1360 GB/s	    1560 GB/s	0.0008821 s 	0.0007691 s 
   axpby	 2746	    1568 GB/s	    1690 GB/s	0.001531 s 	 0.00142 s 
    spmv	  915	    1470 GB/s	    1690 GB/s	0.005714 s 	 0.00497 s 
--------	-----	---------------	---------------	---------------	---------------
   Total	     	               	               	    11.05 s 	    9.856 s
--------	-----	---------------	---------------	---------------	---------------
```

# Example 2: CG solver on the CPU

The hardware used is a node with 2 32-core Intel Sapphire Rapids CPU's and sub-NUMA clustering, giving four NUMA domains with
16 cores each. For simplicity, we run on one NUMA domain (achieving good performance across NUMA domains can be tricky when
interfaceing Python with the RACE library, see below).

We start by setting some environment variables to disable using the GPUs (if any) and using the OpenMP backend 
of Numba:

```bash
export NUMBA_THREADING_LAYER=omp
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=""
```

We use ``taskset`` to restrict threads to one NUMA domain and solve a 5-point Laplace problem with 4M unkowns:

```bash
$ taskset -c 0-15 python3 pcg.py -matgen Laplace2000x2000 -maxit 5000
[...]
number of CG iterations: 4147
relative residual of computed solution: 3.868042e-10
relative error of computed solution: 1.183444e-06
Hardware: cpu (omp, 16 threads)
Hardware assumed for Roofline Model: Intel(R) Xeon(R) Gold 6448Y ("Sapphire Rapids")
(note that the hardware info is taken from [cpu|gpu].json, if does not match your system,
you may want to update those files or delete them to skip the roofline prediction)
Number of threads: 16
--------	-----	---------------	---------------	---------------	---------------
kernel  	calls	 bw_meas       	 bw_roofline   	 t_meas/call   	t_roofline/call
========	=====	===============	===============	===============	===============
     dot	 8295	   110.7 GB/s	     120 GB/s	0.0004334 s 	  0.0004 s 
   axpby	12444	   144.2 GB/s	      98 GB/s	0.0006657 s 	0.0009796 s 
    spmv	 4149	   95.48 GB/s	      98 GB/s	0.003518 s 	0.003428 s 
--------	-----	---------------	---------------	---------------	---------------
   Total	     	               	               	    26.47 s 	    29.73 s
--------	-----	---------------	---------------	---------------	---------------
Total time for CG: 26.7851 seconds.
```

We can now enable a simple polynomial preconditioner and solve the quivalent system

```math
A = I - (L + L^T), k=1
```
```math
L^k A L^{T,k} y = L^k b
```
```math
x = L^{T,k}y
```

This is a very simple method and will typically not lead to faster execution despite reducing the number of CG iterations:

```bash
$ env OMP_NUM_THREADS=16 taskset -c 0-15 python3 pcg.py -matgen Laplace2000x2000 -maxit 5000 -poly_k 1
[...]
number of CG iterations: 1958
relative residual of computed solution: 7.856624e-10
relative error of computed solution: 4.060175e-06
Hardware: cpu (omp, 16 threads)
Hardware assumed for Roofline Model: Intel(R) Xeon(R) Gold 6448Y ("Sapphire Rapids")
(note that the hardware info is taken from [cpu|gpu].json, if does not match your system,
you may want to update those files or delete them to skip the roofline prediction)
Number of threads: 16
--------	-----	---------------	---------------	---------------	---------------
kernel  	calls	 bw_meas       	 bw_roofline   	 t_meas/call   	t_roofline/call
========	=====	===============	===============	===============	===============
     dot	 3917	   124.5 GB/s	     120 GB/s	0.0003854 s 	  0.0004 s 
   axpby	13717	   127.6 GB/s	      98 GB/s	0.0007525 s 	0.0009796 s 
    spmv	 5880	   93.69 GB/s	      98 GB/s	0.002561 s 	0.002448 s 
--------	-----	---------------	---------------	---------------	---------------
   Total	     	               	               	    26.89 s 	     29.4 s
--------	-----	---------------	---------------	---------------	---------------
Total time for constructing precon: 1.09714 seconds.
Total time for solving: 27.1684 seconds.
Total time for CG: 28.2707 seconds.
```

As can be seen, the number of CG iterations is much lower, but the runtime is not. Every application of the triangular factor L is
an additional ``spmv`` operation, so the relative importance of ``spmv`` increases compared to ``dot`` and ``axpby``.

In order to truly benefit from this simple preconditioner, we can use **cache blocking** via the [RACE library](https://github.com/RRZE-HPC/RACE).
The demo framework includes flags ``-c_kernels`` to run with a C/OpenMP backend, and ``-use_RACE`` (implying ``-c_kernels``) to accelerate the polynomial
preconditioner:

```bash
$ env OMP_NUM_THREADS=16 taskset -c 0-15 python3 pcg.py -matgen Laplace2000x2000 -maxit 5000 -poly_k 1 -use_RACE
[...]
number of CG iterations: 1900
relative residual of computed solution: 7.763694e-10
relative error of computed solution: 2.494086e-06
Hardware: cpu (omp, 16 threads)
Hardware assumed for Roofline Model: Intel(R) Xeon(R) Gold 6448Y ("Sapphire Rapids")
(note that the hardware info is taken from [cpu|gpu].json, if does not match your system,
you may want to update those files or delete them to skip the roofline prediction)
Number of threads: 16
--------	-----	---------------	---------------	---------------	---------------
kernel  	calls	 bw_meas       	 bw_roofline   	 t_meas/call   	t_roofline/call
========	=====	===============	===============	===============	===============
     dot	 3801	   96.61 GB/s	     120 GB/s	0.0004968 s 	  0.0004 s 
   axpby	 5707	   127.9 GB/s	      98 GB/s	0.0007506 s 	0.0009796 s 
    spmv	 5706	   149.8 GB/s	      98 GB/s	0.001567 s 	0.002394 s 
--------	-----	---------------	---------------	---------------	---------------
   Total	     	               	               	    15.11 s 	    20.77 s
--------	-----	---------------	---------------	---------------	---------------
Total time for constructing precon: 3.05182 seconds.
Total time for solving: 15.2798 seconds.
Total time for CG: 18.337 seconds.
```

The memory bandwidth reported for the ``spmv`` is higher than what could be delivered from RAM, indicating that 
a substantial part of the traffic now comes from cache instead.

