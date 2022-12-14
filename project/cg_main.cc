#include "cg.hh"
#include <chrono>
#include <iostream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG CGSolver using matrix in the mtx format (Matrix
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

  CGSolver CGSolver;
  CGSolver.read_matrix(argv[1]);
  // CGSolver.generate_lap1d_matrix(64);

  int n = CGSolver.n();
  CGSolver.set_max_iter(n);
  double h = 1. / n;

  CGSolver.init_source_term(h);
  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  auto t1 = clk::now();
  CGSolver.solve(x_d);
  second elapsed = clk::now() - t1;
  std::cout << "Time = " << elapsed.count() << " [s]\n";

  return 0;
}
