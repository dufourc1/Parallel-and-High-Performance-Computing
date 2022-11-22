#ifndef GRID_HH
#define GRID_HH

/* -------------------------------------------------------------------------- */
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
/* -------------------------------------------------------------------------- */

class Grid {
public:
  Grid(int m, int n): m_m(m), m_n(n) {
    cudaMallocManaged(&m_storage, m * n * sizeof(float));
    clear();
  }

  Grid(const Grid & other) :
    m_m{other.m_m}, m_n{other.m_n}, m_storage{other.m_storage}, copy{true} { }

  ~Grid() {
    if (not copy) {
      cudaFree(m_storage);
    }
  }

  /// access the value [i][j] of the grid
  __host__ __device__ inline float & operator()(int i, int j) {
    return m_storage[i * m_n + j];
  }

  __host__ __device__ inline const float & operator()(int i, int j) const {
    return m_storage[i * m_n + j];
  }

  /// set the grid to 0
  __host__ void clear() {
    std::fill(m_storage, m_storage + m_m * m_n, 0.);
  }

  __host__ __device__ int m() const { return m_m; }
  __host__ __device__ int n() const { return m_n; }


  __host__ __device__ float* data() { return m_storage; }
  __host__ __device__ float* data() const { return m_storage; }
private:
  int m_m, m_n;
  float * m_storage;
  bool copy{false};
};

#endif /* GRID_HH */
