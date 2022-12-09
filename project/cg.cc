#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <mpi.h>

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
void CGSolver::solve(std::vector<double> &x, int max_iter)
{
  std::vector<double> r(m_m, 0.0);
  std::vector<double> Ap(m_m, 0.0);
  std::vector<double> tmp(m_m, 0.0);

  std::vector<double> p(m_n, 0.0);

  // r = b - A * x;
  multiply_mat_vector(x, Ap);

  r = m_b;
  // r <- (-1)*Ap + r
  cblas_daxpy(m_m, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  for (int i = 0; i < get_number_rows(); ++i)
  {
    p[i + start_row] = r[i];
  }

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_m, r.data(), 1, r.data(), 1);
  MPI_Allreduce(MPI_IN_PLACE, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // for i = 1:length(b)
  int k = 0;
  if (max_iter == -1 || max_iter > m_m)
  {
    max_iter = m_m;
  }
  for (; k < max_iter; ++k)
  {
    // retrieve p from all the threads
    retrieve_and_concatenate(p);

    if (DEBUG)
    {
      int errors = 0;
      for (int i = 0; i < get_number_rows(); ++i)
      {
        if (p[i + start_row] != r[i])
        {
          errors++;
        }
      }
      if (errors > 0)
      {
        std::cout << "Rank " << rank << " errors: " << errors << std::endl;
      }
    }

    // Ap = A * p;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    // cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
    //            p.data(), 1, 0., Ap.data(), 1);
    multiply_mat_vector(p, Ap);

    // alpha = rsold / (p' * Ap);
    auto w = cblas_ddot(m_m, p.data() + start_row, 1, Ap.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    auto alpha = rsold / std::max(w, rsold * NEARZERO);

    // x = x + alpha * p;
    // only update relevent part of x
    cblas_daxpy(m_m, alpha, p.data() + start_row, 1, x.data() + start_row, 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_m, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_m, r.data(), 1, r.data(), 1);
    MPI_Allreduce(MPI_IN_PLACE, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;

    tmp = r;
    cblas_daxpy(m_m, beta, p.data() + start_row, 1, tmp.data(), 1);
    for (int i = 0; i < get_number_rows(); ++i)
    {
      p[start_row + i] = tmp[i];
    }

    // rsold = rsnew;
    rsold = rsnew;
    if (VERBOSE and k % 100 == 0)
    {
      std::cout << rank << " \t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::endl; // std::flush;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // can be replaced by a gather on the "master" thread
  MPI_Allgatherv(
      // send
      MPI_IN_PLACE,
      // count send,type send
      -1, MPI_DOUBLE,
      // recv buffer
      x.data(),
      // counts, displacements
      counts.data(), displacements.data(),
      MPI_DOUBLE,
      MPI_COMM_WORLD);
  std::vector<double> b(m_n, 0.0);

  if (DEBUG and rank == 0)
  {
    MPI_Allgatherv(
        // send
        m_b.data(),
        // count send,type send
        m_m, MPI_DOUBLE,
        // recv buffer
        b.data(),
        // counts, displacements
        counts.data(), displacements.data(),
        MPI_DOUBLE,
        MPI_COMM_WORLD);
    r.resize(m_n);
    std::fill_n(r.begin(), r.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x.data(), 1, 0., r.data(), 1);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
               std::sqrt(cblas_ddot(m_n, b.data(), 1, b.data(), 1));
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
  counts.resize(size, m_n / size);
  // subset the matrix according to the rank and size
  for (int i = 0; i < m_n % size; ++i)
  {
    counts[i]++;
  }
  displacements.resize(size + 1);
  for (int i = 0; i < size; ++i)
  {
    displacements[i + 1] = displacements[i] + counts[i];
  }
  start_row = displacements[rank];
  end_row = displacements[rank + 1];

  // m_A.subset(start_row, end_row);
  m_m = end_row - start_row;

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
  cblas_dgemv(
      CblasRowMajor,
      CblasNoTrans,
      m_m,
      m_n,
      1.,
      // adjust matrix pointer to start_row
      m_A.data() + start_row * m_n,
      // number of columns is the same
      m_n,
      input.data(),
      1,
      0.,
      output.data(),
      1);
}

void CGSolver::retrieve_and_concatenate(std::vector<double> &x)
{
  MPI_Allgatherv(
      // send
      MPI_IN_PLACE,
      // count send,type send
      -1, MPI_DOUBLE,
      // recv buffer
      x.data(),
      // counts, displacements
      counts.data(), displacements.data(),
      MPI_DOUBLE,
      MPI_COMM_WORLD);
}
