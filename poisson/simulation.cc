/* -------------------------------------------------------------------------- */
#include "simulation.hh"
/* -------------------------------------------------------------------------- */
#include <cmath>
#include <iostream>
#include <vector>
/* -------------------------------------------------------------------------- */
#include <mpi.h>
/* -------------------------------------------------------------------------- */
Simulation::Simulation(int m, int n, int n_iter, int start, int end)
    : number_rows(m), number_columns(n), n_iter(n_iter),
      step_size(1. / n), m_grids(m, n), m_f(m, n), start_index(start),
      end_index(end)
{
}

/* -------------------------------------------------------------------------- */
void Simulation::set_initial_conditions()
{
  for (int i = start_index; i < start_index + number_rows; i++)
  {
    for (int j = 0; j < number_columns; j++)
    {
      m_f(i - start_index, j) = -2. * 100. * M_PI * M_PI * std::sin(10. * M_PI * i * step_size) *
                                std::sin(10. * M_PI * j * step_size);
    }
  }
}

/* -------------------------------------------------------------------------- */
std::tuple<float, int> Simulation::compute(int prank, int psize, bool use_async)
{
  int s = 0;
  float l2 = 0;
  do
  {
    l2 = compute_step();
    if (use_async)
    {
      share_results_asynchrone(prank, psize);
    }
    else
    {
      share_results_synchrone(prank, psize);
    }
    m_grids.swap();

    ++s;
  } while (s < n_iter);

  return std::make_tuple(l2, s);
}

/* -------------------------------------------------------------------------- */
float Simulation::compute_step()
{
  float l2 = 0.;

  Grid &u = m_grids.current();
  Grid &uo = m_grids.old();
  for (int i = 1; i < number_rows - 1; i++)
  {
    for (int j = 1; j < number_columns - 1; j++)
    {
      // computation of the new step
      u(i, j) = 0.25 * (uo(i - 1, j) + uo(i + 1, j) + uo(i, j - 1) +
                        uo(i, j + 1) - m_f(i, j) * step_size * step_size);

      // L2 norm
      l2 += (uo(i, j) - u(i, j)) * (uo(i, j) - u(i, j));
    }
  }
  return l2;
}

/* -------------------------------------------------------------------------- */
void Simulation::share_results_synchrone(int prank, int psize)
{
  int TAG = 0;
  if (prank > 0)
  {
    MPI_Sendrecv(&m_grids.current()(1, 0), number_columns, MPI_FLOAT, prank - 1, TAG,
                 &m_grids.current()(0, 0), number_columns, MPI_FLOAT, prank - 1, TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (prank < psize - 1)
  {
    MPI_Sendrecv(&m_grids.current()(number_rows - 2, 0), number_columns, MPI_FLOAT, prank + 1, TAG,
                 &m_grids.current()(number_rows - 1, 0), number_columns, MPI_FLOAT, prank + 1, TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void Simulation::share_results_asynchrone(int prank, int psize)
{
  int total_number_request = 2;
  if (prank > 0 && prank < psize - 1)
  {
    total_number_request = 4;
  }
  MPI_Request requests[total_number_request];
  MPI_Status status[total_number_request];
  int number_request = 0;
  int TAG = 0;

  if (prank > 0)
  {

    MPI_Isend(&m_grids.current()(1, 0), number_columns, MPI_FLOAT, prank - 1, TAG, MPI_COMM_WORLD, &requests[number_request++]);
    MPI_Irecv(&m_grids.current()(0, 0), number_columns, MPI_FLOAT, prank - 1, TAG, MPI_COMM_WORLD, &requests[number_request++]);
  }
  if (prank < psize - 1)
  {
    MPI_Isend(&m_grids.current()(number_rows - 2, 0), number_columns, MPI_FLOAT, prank + 1, TAG, MPI_COMM_WORLD, &requests[number_request++]);
    MPI_Irecv(&m_grids.current()(number_rows - 1, 0), number_columns, MPI_FLOAT, prank + 1, TAG, MPI_COMM_WORLD, &requests[number_request++]);
  }
  MPI_Waitall(number_request, requests, status);
}