#include <string>
#include <vector>

#ifndef __MATRIX_H_
#define __MATRIX_H_

class Matrix
{
public:
  Matrix(int m = 0, int n = 0) : m_m(m), m_n(n), m_a(m * n) {}
  void resize(int m, int n)
  {
    m_m = m;
    m_n = n;
    m_a.resize(m * n);
  }

  inline double &operator()(int i, int j) { return m_a[i * m_n + j]; }

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }
  inline double *data() { return m_a.data(); }

  void subset(int index_row_start, int index_row_end)
  {
    int m = index_row_end - index_row_start;
    int n = m_n;
    std::vector<double> a(m * n);
    for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        a[i * n + j] = m_a[(i + index_row_start) * n + j];
      }
    }
    m_m = m;
    m_n = n;
    m_a = a;
  }

  void read(const std::string &filename);
  inline void setZero() { std::fill_n(m_a.begin(), m_a.size(), 0.); }

private:
  int m_m{0};
  int m_n{0};
  std::vector<double> m_a;
};

#endif // __MATRIX_H_
