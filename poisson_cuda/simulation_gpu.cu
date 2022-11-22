/* -------------------------------------------------------------------------- */
#include "simulation.hh"
#include "grid.hh"
/* -------------------------------------------------------------------------- */
#include <iostream>
#include <exception>
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
__global__ void compute_step_one_thread_per_row(
    Grid uo, Grid u, Grid f, float h)
{

    // TODO: implement here the 'per row' version.
}

/* -------------------------------------------------------------------------- */
__global__ void compute_step_one_thread_per_entry(
    Grid uo, Grid u, Grid f, float h)
{

    // TODO: implement here the 'per entry' version.
}

/* -------------------------------------------------------------------------- */
void Simulation::compute_step(const dim3 block_size)
{
    Grid &u = m_grids.current();
    Grid &uo = m_grids.old();

    int m = u.m();
    int n = u.n();

    dim3 grid_size; // TODO: define your grid size

    static bool first{true};
    if (first)
    {
        std::cout << "Block size:    " << block_size.x << ":" << block_size.y << "\n"
                  << "Grid_size:     " << grid_size.x << ":" << grid_size.y << std::endl;
        first = false;
    }

#if defined(PER_ROW)
    // TODO: call here the implementation by row
#else
    // TODO: call here the implementation by entry
#endif
    // TODO: did you forget to synchronize ?

    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error("Error Launching Kernel: " + std::string(cudaGetErrorName(error)) + " - " + std::string(cudaGetErrorString(error)));
    }
}
