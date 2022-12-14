#include "cg.hh"
#include "matrix.hh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

const double NEARZERO = 1.0e-14;

// vecVec
#define BLOCK_DIM_VEC 32

// matVec
#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

// ouput = A*x
__global__ void matrix_vector(double *A, double *x, double *output, int n)
{
    // stupid implementation where each thread computes one element of the output
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        double sum = 0;
        for (int j = 0; j < n; j++)
        {
            sum += A[index * n + j] * x[j];
        }
        output[index] = sum;
    }
}

// output = x  + y * scale_y
__global__ void scale_add_vector(double *x, double *y, double *scale_y, double *output, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        output[index] = x[index] + y[index] * *scale_y;
    }
}

// output = x  - y * scale_y
__global__ void scale_subtract_vector(double *x, double *y, double *scale_y, double *output, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        output[index] = x[index] - y[index] * *scale_y;
    }
}

__global__ void diff_vector(double *x, double *y, double *output, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        output[index] = x[index] - y[index];
    }
}

// output = copy(scale)
__global__ void copy_vector(double *x, double *output, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        output[index] = x[index];
    }
}

// output = x/y
__global__ void div_scalar(double *x, double *y, double *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
    {
        *output = *x / max(*y * NEARZERO, *y);
    }
}

// y = copy(x)
__global__ void copy_scalar(double *x, double *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
    {
        *y = *x;
    }
}

void print_gpu_value(double *x, int n)
{
    double *x_host = new double[n];
    cudaMemcpy(x_host, x, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
    {
        std::cout << x_host[i] << " ";
    }
    std::cout << std::endl;
    delete[] x_host;
}

void CGSolver::solve_CUDA(double *A, double *b, double *x)
{
    double r_norm;

    // device memory allocation for vectors
    double *r;
    double *p;
    double *temp;

    cudaMalloc((void **)&r, m_m * sizeof(double));
    cudaMalloc((void **)&p, m_n * sizeof(double));
    cudaMalloc((void **)&temp, m_m * sizeof(double));

    // device memory allocation for scalars
    double *rsold;
    double *rsnew;
    double *alpha;
    double *beta;
    double *scalar_temp;

    cudaMalloc((void **)&rsold, sizeof(double));
    cudaMalloc((void **)&rsnew, sizeof(double));
    cudaMalloc((void **)&alpha, sizeof(double));
    cudaMalloc((void **)&beta, sizeof(double));
    cudaMalloc((void **)&scalar_temp, sizeof(double));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    // 1 thread per row
    // one thread per column
    dim3 matrix_block_size((m_m + 31) / 32);
    dim3 matrix_grid_size((m_n + matrix_block_size.x - 1) / matrix_block_size.x, (m_n + matrix_block_size.y - 1) / matrix_block_size.y);

    // 1 thread per index
    dim3 vector_block_size((m_m + 31) / 32);
    dim3 vector_grid_size((m_n + vector_block_size.x - 1) / vector_block_size.x);

    // r = b - A*x
    matrix_vector<<<matrix_grid_size, matrix_block_size>>>(A, x, temp, m_n);
    diff_vector<<<vector_grid_size, vector_block_size>>>(b, temp, r, m_n);
    copy_vector<<<vector_grid_size, vector_block_size>>>(r, p, m_n);

    cublasDdot(handle, m_n, p, 1, p, 1, rsold);

    int k = 0;
    if (max_iter == -1)
    {
        max_iter = m_n;
    }
    for (; k < max_iter; ++k)
    {
        // temp = A*p
        matrix_vector<<<matrix_grid_size, matrix_block_size>>>(A, p, temp, m_n);

        // alpha = rsold / (p^T temp)
        cublasDdot(handle, m_n, p, 1, temp, 1, scalar_temp);
        div_scalar<<<1, 1>>>(rsold, scalar_temp, alpha);

        // x = x + alpha*p
        scale_add_vector<<<vector_grid_size, vector_block_size>>>(x, p, alpha, x, m_n);

        // r = r - alpha*Ap
        scale_subtract_vector<<<vector_grid_size, vector_block_size>>>(r, temp, alpha, r, m_n);

        // rsnew = r^T r
        cublasDdot(handle, m_n, r, 1, r, 1, rsnew);

        // check convergence
        cudaMemcpy(&r_norm, rsnew, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                  << std::sqrt(r_norm) << "\r" << std::endl;
        if (std::sqrt(r_norm) < m_tolerance)
            break;

        // beta = rsnew / rsold
        div_scalar<<<1, 1>>>(rsnew, rsold, beta);

        // p = r + beta*p
        scale_add_vector<<<vector_grid_size, vector_block_size>>>(r, p, beta, p, m_n);

        // rsold = rsnew
        copy_scalar<<<1, 1>>>(rsnew, rsold);
    }
    std::cout << "Converged in " << k << " iterations. Residual " << std::scientific << std::sqrt(r_norm) << std::endl;
    cublasDdot(handle, m_n, x, 1, x, 1, temp_scalar);
    print_gpu_value(scalar_temp, 1);

    cublasDestroy(handle);

    cudaFree(r);
    cudaFree(p);
    cudaFree(temp);

    cudaFree(rsold);
    cudaFree(rsnew);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(scalar_temp);
}

/*----------------------------------------*/
