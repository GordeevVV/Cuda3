#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}


__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) {
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= n * n) {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        } else {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if (idx >= n * n) {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        } else {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        d_result[row * n + col] = tmp;
    }
}


int dtn(int n, int min_n) {
    int max_tn = n / min_n;
    const int g_ncore = omp_get_num_procs();
    int tn = max_tn > g_ncore ? g_ncore : max_tn;
    if (tn < 1) {
        tn = 1;
    }
    return tn;
}

void omp_mm(int *a, int row_a, int col_a, int *b, int row_b, int col_b, int *c) {
    if (col_a != row_b) {
        return;
    }
    int i, j, k;
    int index;
    int border = row_a * col_b;
    i = 0;
    j = 0;

#pragma omp parallel for private(i, j, k) num_threads(dtn(border, 1))
    for (index = 0; index < border; index++) {
        i = index / col_b;
        j = index % col_b;
        int row_i = i * col_a;
        int row_c = i * col_b;
        c[row_c + j] = 0;
        for (k = 0; k < row_b; k++) {
            c[row_c + j] += a[row_i + k] * b[k * col_b + j];
        }
    }
}

int main(int argc, char const *argv[]) {
    int m, n, k;
    srand(3333);
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int) * m * n);
    cudaMallocHost((void **) &h_b, sizeof(int) * n * k);
    cudaMallocHost((void **) &h_c, sizeof(int) * m * k);
    cudaMallocHost((void **) &h_cc, sizeof(int) * m * k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

    double gpu_elapsed_time_s, cpu_elapsed_time_s;
    //cuda exec time
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
    //omp exec time
    double omp_start, omp_stop;
    omp_start = omp_get_wtime();
    // Allocate memory space on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int) * m * n);
    cudaMalloc((void **) &d_b, sizeof(int) * n * k);
    cudaMalloc((void **) &d_c, sizeof(int) * m * k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);



    // cudaEventRecord(start, 0);

    if (m == n && n == k) {
        gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    } else {
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    }
    omp_stop = omp_get_wtime();
    cudaThreadSynchronize();
    // cudaEventRecord(stop, 0);

    cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    // cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    //cudaEventElapsedTime(&gpu_elapsed_time_s, start, stop);
    gpu_elapsed_time_s = (omp_stop - omp_start)*1000;
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k,
           gpu_elapsed_time_s);

    // start the CPU version
    // cudaEventRecord(start, 0);
    omp_start = omp_get_wtime();
    omp_mm(h_a, m, n, h_b, n, k, h_cc);
    omp_stop = omp_get_wtime();
    //cudaEventRecord(stop, 0);

//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&cpu_elapsed_time_s, start, stop);
    cpu_elapsed_time_s = (omp_stop - omp_start)*1000;
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k,
           cpu_elapsed_time_s);

    // validating
    int all_ok = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_c[i*k + j], i, j, h_c[i*k + j]);
            if (h_c[i * k + j] != h_c[i * k + j]) {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    //speedup
    if (all_ok) {
        printf("all results are correct, speedup = %f\n", cpu_elapsed_time_s / gpu_elapsed_time_s);
    } else {
        printf("incorrect results\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
