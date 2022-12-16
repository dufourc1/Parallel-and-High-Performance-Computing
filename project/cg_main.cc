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
  if (argc < 2 || argc > 6)
  {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename] | optional:  [rows_per_block] [threads_per_line] [max_iter] [N] "
              << std::endl
              << "If N is specified, laplacian will be generated and first matrix will be ignored." << std::endl;
    return 1;
  }

  int rows_per_block = 32;
  int threads_per_row = 32;
  int N = -1;
  int max_iter = -1;
  std::string filename = argv[1];

  if (argc >= 3)
  {
    rows_per_block = atoi(argv[2]);
  }
  if (argc >= 4)
  {
    threads_per_row = atoi(argv[3]);
  }
  if (argc >= 5)
  {
    max_iter = atoi(argv[4]);
  }
  if (argc >= 6)
  {
    N = atoi(argv[5]);
  }

  CGSolver CGSolver(threads_per_row, rows_per_block);
  if (N == -1)
  {
    CGSolver.read_matrix(argv[1]);
  }
  else
  {
    CGSolver.generate_lap1d_matrix(N);
  }

  int n = CGSolver.n();
  if (max_iter == -1)
    max_iter = n;

  CGSolver.set_max_iter(max_iter);
  double h = 1. / n;

  CGSolver.init_source_term(h);
  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  auto t1 = clk::now();
  CGSolver.solve(x_d);
  second elapsed = clk::now() - t1;
  std::cout << elapsed.count() << "," << n << "," << rows_per_block << "," << threads_per_row << "," << max_iter << std::endl;

  return 0;
}
