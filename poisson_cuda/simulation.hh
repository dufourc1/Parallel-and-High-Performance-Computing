#ifndef SIMULATION_HH
#define SIMULATION_HH

/* -------------------------------------------------------------------------- */
#include "double_buffer.hh"
#include "dumpers.hh"
#include "grid.hh"
/* -------------------------------------------------------------------------- */
#include <tuple>
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
class Simulation {
public:
  Simulation(int m, int n);

  /// set the initial conditions, Dirichlet and source term
  virtual void set_initial_conditions();

  /// perform the simulation
  int compute(dim3 block_size = {32, 1});

protected:
  /// compute one step and return an error
  void compute_step(dim3 block_size);
private:
  /// Global problem size
  int m_global_m, m_global_n;

  /// grid spacing
  float m_h_m;
  float m_h_n;

  /// Grids storage
  DoubleBuffer m_grids;

  /// source term
  Grid m_f;

  /// Dumper to use for outputs
  std::unique_ptr<Dumper> m_dumper;
};

#endif /* SIMULATION_HH */
