/* -------------------------------------------------------------------------- */
#include "simulation.hh"
/* -------------------------------------------------------------------------- */
#include <chrono>
#include <iostream>
#include <sstream>
#include <tuple>
#include <chrono>
/* -------------------------------------------------------------------------- */
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */

#define EPSILON 0.005

typedef std::chrono::high_resolution_clock clk;
typedef std::chrono::duration<double> second;

static void usage(const std::string & prog_name) {
  std::cerr << prog_name << " <grid_size> <block_size [default: 32]>" << std::endl;
  exit(0);
}

int main(int argc, char * argv[]) {
  if (argc < 2) usage(argv[0]);

  int N;
  try {
    N = std::stoi(argv[1]);
  } catch(std::invalid_argument &) {
    usage(argv[0]);
  }

  dim3 block_size{32, 1};
  if (argc == 3) {
    try {
      block_size.x = std::stoi(argv[2]);
    } catch(std::invalid_argument &) {
      usage(argv[0]);
    }
  }

  if (argc == 4) {
    try {
      block_size.y = std::stoi(argv[3]);
    } catch(std::invalid_argument &) {
      usage(argv[0]);
    }
  }

  // By default, we use device 0,
  int dev_id = 0;

  cudaDeviceProp device_prop;
  cudaGetDevice(&dev_id);
  cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                 "threads can use ::cudaSetDevice()"
              << std::endl;
    return -1;
  }

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "cudaGetDeviceProperties returned error code " << error
              << ", line(" << __LINE__ << ")" << std::endl;
    return error;
  } else {
    std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
              << "\" with compute capability " << device_prop.major << "."
              << device_prop.minor << std::endl;
  }


  Simulation simu(N, N);

  simu.set_initial_conditions();

  auto start = clk::now();
  int k = simu.compute(block_size);
  auto end = clk::now();

  second time = end - start;

  std::cout << "(" << block_size.x << "x" <<  block_size.y << ") " << N << " "
            << k << " " << std::scientific << " "
            << time.count() << std::endl;

  return 0;
}
