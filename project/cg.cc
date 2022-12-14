#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

/*
    CGSolver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
*/

void CGSolver::solve(std::vector<double> &x)
{

  // device memory allocation for matrix and vectors
  double *A_device;
  double *b_device;
  double *x_device;

  cudaMalloc((void **)&A_device, m_m * m_n * sizeof(double));
  cudaMalloc((void **)&b_device, m_m * sizeof(double));
  cudaMalloc((void **)&x_device, m_n * sizeof(double));

  // copy data from host to device
  cudaMemcpy(A_device, m_A.data(), m_m * m_n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, m_b.data(), m_m * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_device, x.data(), m_n * sizeof(double), cudaMemcpyHostToDevice);

  solve_CUDA(A_device, b_device, x_device);

  cudaDeviceSynchronize();

  // retrieve the solution from device to host
  cudaMemcpy(x.data(), x_device, m_n * sizeof(double), cudaMemcpyDeviceToHost);

  if (DEBUG)
  {
    std::vector<double> r(m_m);
    std::fill_n(r.begin(), r.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x.data(), 1, 0., r.data(), 1);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto rnorm = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1));
    auto res = rnorm / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "residual = " << std::scientific
              << rnorm << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }

  // free device memory
  cudaFree(A_device);
  cudaFree(b_device);
  cudaFree(x_device);
}

void CGSolver::read_matrix(const std::string &filename)
{
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void CGSolver::init_source_term(double h)
{
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++)
  {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}

void CGSolver::generate_lap1d_matrix(int size)
{
  m_A.resize(size, size);
  m_A.setZero();
  m_m = size;
  m_n = size;

  for (int i = 0; i < size; ++i)
  {
    m_A(i, i) = 2;

    if (i > 0)
      m_A(i, i - 1) = -1;
    if (i < size - 1)
      m_A(i, i + 1) = -1;
  }
}