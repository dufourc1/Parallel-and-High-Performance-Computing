#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <numeric>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;
const bool VERBOSE = true;

/*
    cgsolver solves the linear equation A*x = b where A is
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
  std::vector<double> r(m_m);
  std::vector<double> p(m_n);
  std::vector<double> Ap(m_m);
  std::vector<double> tmp(m_n);

  // r = b - A * x;
  std::fill_n(Ap.begin(), Ap.size(), 0.);
  // multiply_mat_vector(x, Ap);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
              x.data(), 1, 0., Ap.data(), 1);

  r = m_b;
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r.data(), 1, p.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    // std::fill_n(Ap.begin(), Ap.size(), 0.);
    // multiply_mat_vector(p, Ap);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                p.data(), 1, 0., Ap.data(), 1);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
                                  rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;

    // rsold = rsnew;
    rsold = rsnew;
    if (VERBOSE)
    {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::endl; // std::flush;
    }
  }

  if (DEBUG and rank == 0)
  {
    std::fill_n(r.begin(), r.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x.data(), 1, 0., r.data(), 1);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}

void CGSolver::read_matrix(const std::string &filename)
{
  // read the full matrix
  m_A.read(filename);
  total_rows = m_A.m();
  m_n = m_A.n();
  if (h == 0.0)
  {
    h = 1.0 / m_n;
  }

  // subset the matrix according to the rank and size
  int number_rows = total_rows / size;
  start_row = rank * number_rows;
  end_row = start_row + number_rows - 1;

  if (rank == size - 1)
  {
    end_row = total_rows - 1;
  }
  // m_A.subset(start_row, end_row);
  m_m = end_row - start_row + 1;

  std::cout << "Rank " << rank << " treats (" << start_row << " x " << end_row << ")" << std::endl;
}

/*
Initialization of the source term b
*/
void Solver::init_source_term()
{
  m_b.resize(m_m);

  for (int i = 0; i < m_m; i++)
  {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}

void CGSolver::init_source_term()
{
  m_b.resize(m_m);

  for (int i = start_row; i < end_row; i++)
  {
    int index_i = i - start_row;
    m_b[index_i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
                   std::sin(10. * M_PI * i * h);
  }
}

/*
output = A[start_row:end_row,:] * input
*/
void CGSolver::multiply_mat_vector(const std::vector<double> &input, std::vector<double> &output)
{
  // multiply our submatrix
  cblas_dgemv(
      CblasRowMajor,
      CblasNoTrans,
      m_m,
      m_n,
      1.,
      // adjust matrix pointer for row_start
      m_A.data() + start_row * m_n,
      // "real" dimension of the matrix
      m_n,
      input.data(),
      1,
      0.,
      output.data(),
      1);
}
