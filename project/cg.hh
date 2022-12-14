#include "matrix.hh"
#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>

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
  void read_matrix(const std::string &filename);
  void init_source_term(double h);
  void solve(std::vector<double> &x);

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  void tolerance(double tolerance) { m_tolerance = tolerance; }

protected:
  int m_m{0};
  int m_n{0};
  std::vector<double> m_b;
  double m_tolerance{1e-10};

private:
  Matrix m_A;
};

#endif /* __CG_HH__ */
