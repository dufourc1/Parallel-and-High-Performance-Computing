#include "matrix.hh"
#include "matrix_coo.hh"
#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef __CG_HH__
#define __CG_HH__

/*
CGSolver::solve(std::vector<double> & x)
CGSolver::read_matrix(const std::string & filename)
CGSolver::init_source_term(int n, double h)
*/
class CGSolver
{
public:
  CGSolver(int n_threads_per_row, int n_rows_per_block)
  {
    this->threads_per_row = n_threads_per_row;
    this->rows_per_block = n_rows_per_block;
  };
  void read_matrix(const std::string &filename);
  void init_source_term(double h);
  void solve(std::vector<double> &x);
  inline int m() const { return m_m; }
  inline int n() const { return m_n; }
  void tolerance(double tolerance) { m_tolerance = tolerance; }
  void generate_lap1d_matrix(int size);
  void set_max_iter(int number) { max_iter = number; }

protected:
  int threads_per_row;
  int rows_per_block;
  int m_m{0};
  int m_n{0};
  std::vector<double> m_b;
  double m_tolerance{1e-10};
  void solve_CUDA(double *A, double *b, double *x);
  int max_iter{-1};

private:
  Matrix m_A;
};

#endif /* __CG_HH__ */
