#include "cg.hh"
#include "matrix.hh"
#include <cuda_runtime.h>

__global__ void do_matrix_multiplication(const double *A, int n, const double *input, double *output)
{
    // extern __shared__ double row_sums[];
    //  suffit d'utiliser la memoire
}