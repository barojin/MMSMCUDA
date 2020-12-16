#include <stdio.h>

typedef struct {
    int n;
    int m;
    int tile;
    float* arr;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

void printa(float *A, int n, int m);
void generateMatrix(float *A, int n, int m, int num);
__global__ void MulKernel(const Matrix, const Matrix, Matrix);
__global__ void MulKernelShared(const Matrix, const Matrix, Matrix);

__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.arr[row * A.tile + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.arr[row * A.tile + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.n    = BLOCK_SIZE;
    Asub.m   = BLOCK_SIZE;
    Asub.tile   = A.tile;
    Asub.arr = &A.arr[A.tile * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatrixMultiplication()
__global__ void MulKernel(Matrix A, Matrix B, Matrix C)
{
    float sum = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < A.n; ++e)
        sum += A.arr[row * A.n + e] * B.arr[e * B.n + col];
    C.arr[row * C.n + col] = sum;
}

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
__host__ void MatrixMultiplication(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.n = A.n; d_A.m = A.m;
    size_t size = A.n * A.m * sizeof(float);
    cudaMalloc(&d_A.arr, size);
    cudaMemcpy(d_A.arr, A.arr, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.n = B.n; d_B.m = B.m;
    size = B.n * B.m * sizeof(float);
    cudaMalloc(&d_B.arr, size);
    cudaMemcpy(d_B.arr, B.arr, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.n = C.n; d_C.m = C.m;
    size = C.n * C.m * sizeof(float);
    cudaMalloc(&d_C.arr, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.n / dimBlock.x, A.m / dimBlock.y);
    MulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.arr, d_C.arr, size, cudaMemcpyDeviceToHost);

    // printa(C.arr, C.n, C.m);
    cudaFree(d_A.arr);
    cudaFree(d_B.arr);
    cudaFree(d_C.arr);
}

__global__ void MulKernelShared(Matrix A, Matrix B, Matrix C)
{
    // Use the block size of subarr of Matrix C.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    float sum = 0;
    for (int m = 0; m < (A.n / BLOCK_SIZE); ++m) {

        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // subarr A and B are stored in Shared memory
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize the storing data Asub and B sub into As and Bs.
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            sum += As[row][e] * Bs[e][col];

        // Synchronize to block the new generation of Asub and Bsub during iteration.
        __syncthreads();
    }

    SetElement(Csub, row, col, sum);
}

__host__ void MatrixMultiplicationShared(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.n = d_A.tile = A.n; d_A.m = A.m;
    size_t size = A.n * A.m * sizeof(float);
    cudaMalloc(&d_A.arr, size);
    cudaMemcpy(d_A.arr, A.arr, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.n = d_B.tile = B.n; d_B.m = B.m;
    size = B.n * B.m * sizeof(float);
    cudaMalloc(&d_B.arr, size);
    cudaMemcpy(d_B.arr, B.arr, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.n = d_C.tile = C.n; d_C.m = C.m;
    size = C.n * C.m * sizeof(float);
    cudaMalloc(&d_C.arr, size);

    // dim3(uint3 x, uint3 y), specify demensions. default is 1.
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 16 x 16 , dimBlock.x * dimBlock.y, total 256 threads
    // printf("dimBlock.x: %d, dim.y: %d\n", dimBlock.x, dimBlock.y);
    dim3 dimGrid(B.n / dimBlock.x, A.m / dimBlock.y);
    MulKernelShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.arr, d_C.arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.arr);
    cudaFree(d_B.arr);
    cudaFree(d_C.arr);
}

// print the float array
void printa(float *A, int n, int m){
    for (int i=0; i<n*m; i++){
        printf("%.f ", A[i]);
    }
    printf("\n");
}

// fill the number in float array
void generateMatrix(float *A, int n, int m, int num){
    for (int i=0; i<n*m; i++){
        A[i] = num;
    }
}

void generateMatrix2d(float **a, int row, int col, int num){
    a = (float **)calloc(row, sizeof(float*));
    for(int i = 0; i < row; i++)
        a[i] = (float *) calloc (col, sizeof(float));

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            a[i][j] = num;
        }
    }
}

void MatrixMultiplicationCPU(float **a, float **b, float **c, int n, int m){
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            for(int k = 0; k < n; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
}

int main(int argc, char const *argv[]) {
    int n, w, m;
    float ms = 0; // milliseconds
    float **a, **b, **c;
    int num, row, col;
    size_t sizeA, sizeB, sizeC;
    float *Ae, *Be, *Ce;

    for (int i= 32384; i >= 128; i >>= 1){
        // n = m = w = i;
        n = m  = i;
        w = i / 2;
        printf("N x N = %d \n", m * n);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
//
//        num = 2, row = n, col = w;
//
//        a = (float **)calloc(row, sizeof(float*));
//        for(int i = 0; i < row; i++)
//            a[i] = (float *) calloc (col, sizeof(float));
//
//        for(int i = 0; i < row; i++){
//            for(int j = 0; j < col; j++){
//                a[i][j] = num;
//            }
//        }
//
//        num = 3, row = w, col = m;
//        b = (float **)calloc(row, sizeof(float*));
//        for(int i = 0; i < row; i++)
//            b[i] = (float *) calloc (col, sizeof(float));
//
//        for(int i = 0; i < row; i++){
//            for(int j = 0; j < col; j++){
//                b[i][j] = num;
//            }
//        }
//
//        num = 0, row = n, col = m;
//        c = (float **)calloc(row, sizeof(float*));
//        for(int i = 0; i < row; i++)
//            c[i] = (float *) calloc (col, sizeof(float));
//
//        for(int i = 0; i < row; i++){
//            for(int j = 0; j < col; j++){
//                c[i][j] = num;
//            }
//        }
////    generateMatrix2d(a, n, w, 2);
////    generateMatrix2d(b, w, m, 3);
////    generateMatrix2d(a, n, m, 0);
//
//        cudaEventRecord(start);
//        // Matrix Multiplication on CPU, no parallel
//        for(int i = 0; i < n; ++i)
//            for(int j = 0; j < m; ++j)
//                for(int k = 0; k < n; ++k)
//                {
//                    c[i][j] += a[i][k] * b[k][j];
//                }
//
//        cudaEventRecord(stop);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&ms, start, stop);
//        printf("CPU Multiplication time: %fn(ms)\n", ms);

        sizeA = m * w * sizeof(float);
        sizeB = w * n * sizeof(float);
        sizeC = m * n * sizeof(float);

        Ae = (float*) malloc(sizeA);
        Be = (float*) malloc(sizeB);
        Ce = (float*) malloc(sizeC);

        Matrix A = {n, n, w, Ae};
        Matrix B = {w, w, m, Be};
        Matrix C = {n, n, m, Ce};

        generateMatrix(A.arr, A.n, A.m, 2);
        generateMatrix(B.arr, B.n, B.m, 3);
        generateMatrix(C.arr, C.n, C.m, 0);

        cudaEventRecord(start);
        // Matrix Multiplication without shared memory
        MatrixMultiplication(B, A, C);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("Matrix Multiplication time: %fn(ms)\n", ms);

        cudaEventRecord(start);
        // Matrix Multiplication with shared memory
        MatrixMultiplicationShared(B, A, C);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf("Matrix Multiplication shared time: %fn(ms)\n", ms);

        free(a); free(b); free(c); free(Ae); free(Be); free(Ce);
    }


    return 0;
}
