#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <math.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char **argv)
{

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2)
  {
    if (rank == 0)
    {
      std::cerr << "Usage: " << argv[0] << " [martix-market-filename] or [matrix-number-rows]"
                << std::endl;
    }
    return 1;
  }

  CGSolver solver(rank, size);
  int max_iter = 100;
  int n = -1;

  if (argc == 2)
  {
    solver.read_matrix(argv[1]);
    solver.split_work();
    max_iter = solver.n();
  }
  else if (argc == 3)
  {
    int n = atoi(argv[2]); // floor(sqrt(size) * atoi(argv[2]));
    solver.split_work(n);
    solver.generate_lap1d_matrix(n);
    max_iter = std::min(max_iter, n);
  }
  else
  {
    if (rank == 0)
    {
      std::cerr << "Usage: " << argv[0] << " [martix-market-filename] or [matrix-number-rows] | [N]"
                << std::endl;
    }
    return 1;
  }

  n = solver.n();

  solver.init_source_term();

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  auto t1 = clk::now();
  solver.solve(x_d, max_iter);

  second elapsed = clk::now() - t1;
  if (solver.get_rank() == 0)
  {
    std::cout << solver.get_size() << "," << n << "," << elapsed.count() << std::endl;
  }

  return 0;
}
