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
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if (i < uo.m() - 1)
    {
        for (int j = 1; j < uo.n() - 1; j++)
        {
            u(i, j) = 0.25 * (uo(i - 1, j) + uo(i + 1, j) + uo(i, j - 1) +
                              uo(i, j + 1) - f(i, j) * h * h);
        }
    }
}

/* -------------------------------------------------------------------------- */
__global__ void compute_step_one_thread_per_entry(
    Grid uo, Grid u, Grid f, float h)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    if (i < uo.m() - 1 && j < uo.n() - 1)
    {
        u(i, j) = 0.25 * (uo(i - 1, j) + uo(i + 1, j) + uo(i, j - 1) +
                          uo(i, j + 1) - f(i, j) * h * h);
    }
}

/* -------------------------------------------------------------------------- */
void Simulation::compute_step(const dim3 block_size)
{
    Grid &u = m_grids.current();
    Grid &uo = m_grids.old();

    int m = u.m();
    int n = u.n();

#if defined(PER_ROW)
    int grid_size_y = 1;

#else
    int grid_size_y = (m - 2 + block_size.y - 1) / block_size.y;

#endif

    dim3 grid_size((n - 2 + block_size.x - 1) / block_size.x, grid_size_y); // only works for per row

    static bool first{false};
    if (first)
    {
        std::cout << "Block size:    " << block_size.x << ":" << block_size.y << "\n"
                  << "Grid_size:     " << grid_size.x << ":" << grid_size.y << std::endl;
        first = false;
    }

#if defined(PER_ROW)
    // TODO: call here the implementation by row
    compute_step_one_thread_per_row<<<grid_size, block_size>>>(uo, u, m_f, 1. / n);
#else
    compute_step_one_thread_per_entry<<<grid_size, block_size>>>(uo, u, m_f, 1. / n);
// TODO: call here the implementation by entry
#endif
    cudaDeviceSynchronize();

    auto error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error("Error Launching Kernel: " + std::string(cudaGetErrorName(error)) + " - " + std::string(cudaGetErrorString(error)));
    }
}
