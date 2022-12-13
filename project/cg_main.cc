#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <sstream>

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

  // int number_processes = atoi(argv[1]);
  // solver.generate_lap1d_matrix(number_processes * 1000);

  solver.read_matrix(argv[1]);
  solver.split_work();

  int n = solver.n();

  solver.init_source_term();
  int max_iter = 500;

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
