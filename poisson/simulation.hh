#ifndef SIMULATION_HH
#define SIMULATION_HH

/* -------------------------------------------------------------------------- */
#include "double_buffer.hh"
#include "dumpers.hh"
#include "grid.hh"
/* -------------------------------------------------------------------------- */
#include <tuple>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
class Simulation
{
public:
  Simulation(int m, int n, int n_iter, int start, int end);

  /// set the initial conditions, Dirichlet and source term
  virtual void set_initial_conditions();

  /// perform the simulation
  std::tuple<float, int> compute(int prank, int psize, bool use_async);

protected:
  /// compute one step and return an error
  virtual float compute_step();

  /// share the results between the processes
  virtual void share_results_synchrone(int prank, int psize);
  virtual void share_results_asynchrone(int prank, int psize);

private:
  /// Global problem size
  int number_rows, number_columns;

  /// Number of iterations
  int n_iter;

  /// grid spacing
  float step_size;

  /// Grids storage
  DoubleBuffer m_grids;

  /// source term
  Grid m_f;

  /// start and end index of the rows
  int start_index, end_index;
};

#endif /* SIMULATION_HH */
