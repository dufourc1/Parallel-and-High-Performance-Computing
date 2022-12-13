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
  virtual void solve(std::vector<double> &x, int max_iter) = 0;

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
  // constructors
  CGSolver(int rank, int size) : Solver(rank, size) {}
  CGSolver(int rank, int size, double h) : Solver(rank, size, h) {}

  // from parent class
  virtual void read_matrix(const std::string &filename);
  virtual void solve(std::vector<double> &x, int max_iter);
  void init_source_term();

  // linear algebra helpers
  void multiply_mat_vector(const std::vector<double> &input, std::vector<double> &output);

  // getters
  int get_number_rows() const { return end_row - start_row; }
  int get_start() { return start_row; }
  int get_end() { return end_row; }

  // mpi helpers
  std::vector<int> get_counts() { return counts; }
  std::vector<int> get_displacements() { return displacements; }
  void retrieve_and_concatenate(std::vector<double> &x);
  void generate_lap1d_matrix(int size);

private:
  Matrix m_A;
  int total_rows{0};
  int start_row{0}; // inclusive
  int end_row{0};   // exclusive
  std::vector<int> counts;
  std::vector<int> displacements;
};

#endif /* __CG_HH__ */
