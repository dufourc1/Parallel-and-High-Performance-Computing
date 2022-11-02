/* -------------------------------------------------------------------------- */
#include "simulation.hh"
/* -------------------------------------------------------------------------- */
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <tuple>
#include <chrono>
/* -------------------------------------------------------------------------- */
#include <mpi.h>
/* -------------------------------------------------------------------------- */

typedef std::chrono::high_resolution_clock clk;
typedef std::chrono::duration<double> second;

static void usage(const std::string &prog_name)
{
  std::cerr << prog_name << " <grid_size> <n_iter> optional <use_async>" << std::endl;
  exit(0);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int prank, psize;

  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  int N;
  int n_iter;
  bool use_async = false;
  if (argc != 3 && argc != 4)
    usage(argv[0]);
  if (argc == 3)
  {
    std::stringstream arg_0(argv[1]), arg_1(argv[2]);

    arg_0 >> N;
    arg_1 >> n_iter;
    if (arg_0.fail() || arg_1.fail())
      usage(argv[0]);
  }
  else
  {
    std::stringstream arg_0(argv[1]), arg_1(argv[2]), arg_2(argv[3]);
    arg_0 >> N;
    arg_1 >> n_iter;
    arg_2 >> use_async;
    if (arg_0.fail() || arg_1.fail() || arg_2.fail())
      usage(argv[0]);
  }
  if (prank == 0)
  {
    std::cout << "Number of processes: " << psize << ", number of points: " << N << ", number of iterations " << n_iter << std::endl;
  }
  /// divide the number of rows into psize parts of approximately equal size
  int rows = (N + psize - 1) / psize;
  /// add  ghost rows depending on where they are needed
  int row_start = std::max(0, prank * rows - 1);
  int row_end = std::min(N - 1, (prank + 1) * rows);
  /// update the number of rows
  rows = row_end - row_start + 1;

  Simulation simu(rows, N, n_iter, row_start, row_end);

  simu.set_initial_conditions();

  float l2;
  float l2_global;
  int k;

  auto start = clk::now();
  std::tie(l2, k) = simu.compute(prank, psize, use_async);
  MPI_Reduce(&l2, &l2_global, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  auto end = clk::now();

  second time = end - start;

  if (prank == 0)
  {
    std::cout << psize << " " << N << " "
              << k << " " << std::scientific << l2_global << " "
              << time.count() << std::endl;
    // save results to a file
    std::ofstream outfile;
    outfile.open("output/results.txt", std::ios_base::app);
    outfile << psize << "," << N << ","
            << k << "," << l2_global << ","
            << time.count() << "," << use_async << "\n";
  }

  MPI_Finalize();

  return 0;
}
