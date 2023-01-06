#include "cg.hh"
#include "matrix.hh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cblas.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;
const bool DEBUG_Indices = false;
const bool VERBOSE = false;

__global__ void check_indices(double *A, double *x, double *output, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (blockIdx.y < 2)
    {
        printf("thread_index (%d,%d), block index (%d,%d) -> row: %d, tid: %d, global index: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
               row, tid, threadId);
    }
}

// ouput = A*x
// only work for one dimensional grid: each row is processed by one block at most
__global__ void matrix_vector(double *A, double *x, double *output, int n)
{
    // stupid implementation where each thread computes one element of the output
    extern __shared__ double row_sums[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + blockDim.x * threadIdx.y;

    if (row < n)
    {

        double sum = 0;
        // blockIdx.y should be 1 for this kernel to work
        for (int j = threadIdx.x; j < n; j += blockDim.x)
        {
            sum += A[row * n + j] * x[j];
        }

        // store result in shared memory
        row_sums[tid] = sum;

        // wait for all threads in the block to finish and then aggregate the results
        __syncthreads();

        // jmp >>= 1 is equivalent to jmp /= 2
        for (int jmp = blockDim.x / 2; jmp > 0; jmp >>= 1)
        {
            // first iteration, each thread in the first half of the block adds the result of the second half to its result
            if (threadIdx.x < jmp)
            {
                row_sums[tid] += row_sums[tid + jmp];
            }
            __syncthreads();
        }

        // first thread in the row writes the result to global memory since it aggregated all the results
        if (threadIdx.x == 0)
        {
            output[row] = row_sums[tid];
        }
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

    // shared memory for matrix vector multiplication
    int shared_mem = rows_per_block * threads_per_row * sizeof(double);
    dim3 matrix_block_size(threads_per_row, rows_per_block);
    // one row is assigned to one block (not more) →  only a one dimensional grid
    dim3 matrix_grid_size(1, (m_n + matrix_block_size.y - 1) / matrix_block_size.y);

    // 1 thread per row for vector operations
    // old -> dim3 vector_block_size(threads_per_row);
    dim3 vector_block_size(1024);
    dim3 vector_grid_size((m_n + vector_block_size.x - 1) / vector_block_size.x);

    if (DEBUG and VERBOSE)
    {
        std::cout << "matrix_block_size: " << matrix_block_size.x << " " << matrix_block_size.y << std::endl;
        std::cout << "matrix_grid_size: " << matrix_grid_size.x << " " << matrix_grid_size.y << std::endl;
    }

    if (DEBUG_Indices)
    {
        check_indices<<<matrix_grid_size, matrix_block_size>>>(A, p, temp, m_n);
    }

    // temp = A*x
    matrix_vector<<<matrix_grid_size, matrix_block_size, shared_mem>>>(A, x, temp, m_n);
    // r = b - A*x
    diff_vector<<<vector_grid_size, vector_block_size>>>(b, temp, r, m_n);
    cublasDdot(handle, m_n, r, 1, r, 1, rsold);

    // p = r
    copy_vector<<<vector_grid_size, vector_block_size>>>(r, p, m_n);

    /*
   We don't need cudaDeviceSynchronize() here because all the kernels launched by the same stream are executed sequentially.
   cudaMemcpy() is a blocking call, so we don't need to synchronize either.
   */

    int k = 0;
    if (max_iter == -1)
    {
        max_iter = m_n;
    }
    for (; k < max_iter; ++k)
    {
        // temp = A*p
        matrix_vector<<<matrix_grid_size, matrix_block_size, shared_mem>>>(A, p, temp, m_n);

        // scalar_temp = p^T temp
        cublasDdot(handle, m_n, p, 1, temp, 1, scalar_temp);
        // alpha = rsold / scalar_temp
        div_scalar<<<1, 1>>>(rsold, scalar_temp, alpha);

        // x = x + alpha*p
        scale_add_vector<<<vector_grid_size, vector_block_size>>>(x, p, alpha, x, m_n);

        // r = r - alpha*Ap
        scale_subtract_vector<<<vector_grid_size, vector_block_size>>>(r, temp, alpha, r, m_n);

        // rsnew = r^T r
        cublasDdot(handle, m_n, r, 1, r, 1, rsnew);

        // check convergence
        // copy rsnew to host to do the check
        cudaMemcpy(&r_norm, rsnew, sizeof(double), cudaMemcpyDeviceToHost);

        if (DEBUG && k % 100 == 0 && VERBOSE)
        {
            std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                      << std::sqrt(r_norm) << "\r" << std::endl;
        }

        if (std::sqrt(r_norm) < m_tolerance)
        {
            break;
        }

        // beta = rsnew / rsold
        div_scalar<<<1, 1>>>(rsnew, rsold, beta);

        // p = r + beta*p
        scale_add_vector<<<vector_grid_size, vector_block_size>>>(r, p, beta, p, m_n);

        // rsold = rsnew
        copy_scalar<<<1, 1>>>(rsnew, rsold);
    }
    if (DEBUG)
    {
        std::cout << "Converged in " << k << " iterations. ||r|| = " << std::scientific << std::sqrt(r_norm) << std::endl;
    }

    // free device memory
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
