/*******************************************************************************************/
/* This file is part of the training material available at                                 */
/* https://github.com/jthies/PELS                                                          */
/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
/* included in this software.                                                              */
/*                                                                                         */
/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
/*                                                                                         */
/*******************************************************************************************/

#include <sys/types.h>
#include <stdio.h>
#include <math.h>

// Basic C implemenattions of axpy and dot using OpenMP.

// copy y = x with multiple threads
void copy_vector(size_t N, double *restrict X, double *restrict Y)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++)
  {
    Y[i]=X[i];
  }
}

// Operation that loads two and writes one vector (z=a*x+y).
void axpby(size_t N, double a, const double *restrict X, double b, double *restrict Y)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++)
  {
    Y[i]=a*X[i]+b*Y[i];
  }
}

void init(size_t N, double *restrict X, double val)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++)
  {
    X[i]=val;
  }
}

// vector scaling y[i] = v[i]*x[i]
void vscale(size_t N, double const *restrict V, double const *restrict X, double *restrict Y)
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; i++)
  {
    Y[i]=V[i]*X[i];
  }
}

double dot(size_t N, const double* X, const double* Y)
{
  double dot=0.0;
#pragma omp parallel for schedule(static) reduction(+:dot)
  for (size_t i=0; i<N; i++) dot+=X[i]*Y[i];
  return dot;
}

