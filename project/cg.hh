#include "matrix.hh"
#include <cblas.h>
#include <string>
#include <vector>

#ifndef __CG_HH__
#define __CG_HH__

class Solver
{
public:
  Solver(int rank, int size) : rank(rank), size(size) {}
  Solver(int rank, int size, double h) : rank(rank), size(size), h(h) {}
  virtual void read_matrix(const std::string &filename) = 0;
  inline int get_rank() const { return rank; };
  inline int get_size() const { return size; };
  void init_source_term();
  virtual void solve(std::vector<double> &x) = 0;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  void tolerance(double tolerance) { m_tolerance = tolerance; }

protected:
  int m_m{0};
  int m_n{0};
  int rank;
  int size;
  double h{0.0};
  std::vector<double> m_b;
  double m_tolerance{1e-10};
};

class CGSolver : public Solver
{
public:
  CGSolver(int rank, int size) : Solver(rank, size) {}
  CGSolver(int rank, int size, double h) : Solver(rank, size, h) {}
  virtual void read_matrix(const std::string &filename);
  virtual void solve(std::vector<double> &x);
  void multiply_mat_vector(const std::vector<double> &input, std::vector<double> &output);
  int get_number_rows() const { return end_row - start_row + 1; }
  void init_source_term();

private:
  Matrix m_A;
  int total_rows{0};
  int start_row{0};
  int end_row{0};
};

#endif /* __CG_HH__ */
