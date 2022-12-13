#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Testing the performance of the CG solver
*/
void test_solver(CGSolver &solver, const std::string &filename, int max_iter)
{
  solver.read_matrix(filename);

  int n = solver.n();

  solver.init_source_term();
  if (max_iter == -1 || max_iter > n)
  {
    max_iter = n;
  }

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  auto t1 = clk::now();
  solver.solve(x_d, max_iter);

  second elapsed = clk::now() - t1;
  if (solver.get_rank() == 0)
  {
    std::cout << solver.get_size() << "," << n << "," << elapsed.count() << std::endl;
  }
}

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
      std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
                << std::endl;
    }
    return 1;
  }

  CGSolver solver(rank, size);
  test_solver(solver, argv[1], -1);

  return 0;
}
