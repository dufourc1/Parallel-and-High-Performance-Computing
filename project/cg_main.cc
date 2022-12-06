#include "cg.hh"
#include <chrono>
#include <iostream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Testing the performance of the CG solver
*/
void test_solver(Solver &solver, const std::string &filename)
{
  solver.read_matrix(filename);

  int n = solver.n();
  int m = solver.m();
  double h = 1. / n;

  solver.init_source_term(h);

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);
  std::string solver_name = typeid(solver).name();

  std::cout << "Call " << solver_name << " on matrix size (" << m << " x " << n << ")"
            << std::endl;
  auto t1 = clk::now();
  solver.solve(x_d);
  second elapsed = clk::now() - t1;
  std::cout << "Time for " << solver_name << "  = " << elapsed.count() << " [s]\n";
}

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
              << std::endl;
    return 1;
  }

  CGSolver solver;
  test_solver(solver, argv[1]);

  CGSolverSparse solver_sparse;
  test_solver(solver_sparse, argv[1]);

  return 0;
}
