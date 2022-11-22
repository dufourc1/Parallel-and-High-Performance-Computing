/* -------------------------------------------------------------------------- */
#include "simulation.hh"
/* -------------------------------------------------------------------------- */
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
Simulation::Simulation(int m, int n)
    : m_global_m(m), m_global_n(n), m_h_m(1. / m),
      m_h_n(1. / n), m_grids(m, n), m_f(m, n),
      m_dumper(new DumperBinary(m_grids.old())) {}

/* -------------------------------------------------------------------------- */
void Simulation::set_initial_conditions()
{
  for (int i = 0; i < m_global_m; i++)
  {
    for (int j = 0; j < m_global_n; j++)
    {
      m_f(i, j) = -2. * 100. * M_PI * M_PI * std::sin(10. * M_PI * i * m_h_m) *
                  std::sin(10. * M_PI * j * m_h_n);
    }
  }
}

/* -------------------------------------------------------------------------- */
int Simulation::compute(dim3 block_size)
{
  int s;

  for (s = 0; s < 1000; ++s)
  {
    compute_step(block_size);
    m_grids.swap();
  }

#if defined(DEBUG)
  auto &u = m_grids.current();
  auto &uo = m_grids.old();
  float l2 = 0.;
  for (int i = 0; i < u.m(); ++i)
  {
    for (int j = 0; j < u.n(); ++j)
    {
      l2 += (u(i, j) - uo(i, j)) * (u(i, j) - uo(i, j));
    }
  }

  std::cout << "l2 norm: " << std::sqrt(l2) << std::endl;
#endif
  // m_dumper->dump(s);
  return s;
}
