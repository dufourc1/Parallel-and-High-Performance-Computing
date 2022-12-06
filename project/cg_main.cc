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
void test_solver(Solver &solver, const std::string &filename)
{
  solver.read_matrix(filename);

  int n = solver.n();
  int m = solver.m();

  solver.init_source_term();

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  std::cout << "CG " << solver.get_rank() << " on matrix size (" << m << " x " << n << ")"
            << std::endl;
  auto t1 = clk::now();
  solver.solve(x_d);
  /*
  if (solver.get_rank() == 0)
  {
    std::vector<double> buffer(n);
    MPI_Gather(&x_d, m, MPI_FLOAT, &buffer, m, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }
  */

  second elapsed = clk::now() - t1;
  std::cout << "Time for " << solver.get_rank() << "  = " << elapsed.count() << " [s]\n";
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
  if (rank == 0)
  {
    std::cout << "Testing CG solver on " << size << " processes." << std::endl;
  }

  CGSolver solver(rank, size);
  test_solver(solver, argv[1]);

  return 0;
}
