#include <stdio.h>
#include <cuda/cmath>
#include <ctime>
#include <chrono>
#include <algorithm>
#include "include/cuda_utils.hpp"

__global__ void cuda_matrix_multiply_shared(const int* matrix_a, const int* matrix_b, int* matrix_c, const int M, const int K, const int N) {
      int row = threadIdx.y + blockIdx.y * blockDim.y;
      int col = threadIdx.x + blockIdx.x * blockDim.x;

      // Un único array dinámico, partido en dos mitades
      extern __shared__ int shared_mem[];
      int* tileA = shared_mem;                              // [TILE x TILE]
      int* tileB = shared_mem + blockDim.y * blockDim.x;   // [TILE x TILE]

      int sum = 0;
      int numTiles = (K + blockDim.x - 1) / blockDim.x;

      for (int t = 0; t < numTiles; t++) {
          // Carga cooperativa — cada hilo carga un elemento
          int aCol = t * blockDim.x + threadIdx.x;
          int bRow = t * blockDim.y + threadIdx.y;

          tileA[threadIdx.y * blockDim.x + threadIdx.x] = (row < M && aCol < K)
              ? matrix_a[row * K + aCol] : 0;

          tileB[threadIdx.y * blockDim.x + threadIdx.x] = (bRow < K && col < N)
              ? matrix_b[bRow * N + col] : 0;

          __syncthreads();  // esperar a que el tile esté listo

          for (int i = 0; i < blockDim.x; i++)
              sum += tileA[threadIdx.y * blockDim.x + i] * tileB[i * blockDim.x + threadIdx.x];

          __syncthreads();  // antes de sobreescribir el tile en la siguiente iteración
      }

      if (row < M && col < N)
          matrix_c[row * N + col] = sum;
  }

__global__ void cuda_matrix_multiply_global(const int* matrix_a, const int* matrix_b, int* matrix_c, const int M, const int K, const int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row >= M || col >= N) return;

    int sum = 0;
    for(int i=0; i<K; i++){
        sum += matrix_a[row * K + i] * matrix_b[i * N + col];
    }

    matrix_c[row * N + col] = sum; 
}

void serial_matrix_multiply(const int* matrix_a, const int* matrix_b, int* matrix_c, const int M, const int K, const int N){
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int i = 0; i < K; i++) {
                sum += matrix_a[row * K + i] * matrix_b[i * N + col];
            }
            matrix_c[row * N + col] = sum;
        }
    }
}

void init_array(int* Array, const int &size){
    for(int i=0; i<size; i++)
        Array[i] = rand() % 100;
}

void print_matrix_10x10(int* matrix, int rows, int cols, const char* name){
    if(rows > 10) rows = 10;
    if(cols > 10) cols = 10;
    printf("%s\n", name);
    for(int i=0; i<rows; i++){
        printf("| ");
        for(int j=0; j<cols; j++){
            printf("%5d ", matrix[j + i * cols]);
        }
        printf("|\n");
    }
    printf("\n");
}

void matrix_multiply(const int M, const int K, const int N, const bool serial){
    int* A = nullptr;
    int* B = nullptr;
    int* C_global = nullptr;
    int* C_shared = nullptr;

    int* devA = nullptr;
    int* devB = nullptr;
    int* devC = nullptr;

    cudaMallocHost(&A, M*K*sizeof(int));
    cudaMallocHost(&B, K*N*sizeof(int));
    cudaMallocHost(&C_global, M*N*sizeof(int));
    cudaMallocHost(&C_shared, M*N*sizeof(int));

    init_array(A, M*K);
    init_array(B, K*N);

    cudaMalloc(&devA, M*K*sizeof(int));
    cudaMalloc(&devB, K*N*sizeof(int));
    cudaMalloc(&devC, M*N*sizeof(int));

    cudaMemcpy(devA, A, M*K*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, K*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(devC, 0, M*N*sizeof(int));

    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x -1) / blockSize.x,
                  (M + blockSize.y -1) / blockSize.y);  

    float ms_global = measure_cuda_time([&]() {
        cuda_matrix_multiply_global<<<gridSize, blockSize>>>(devA, devB, devC, M, K, N);
    });

    cudaMemcpy(C_global, devC, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemset(devC, 0, M*N*sizeof(int));

    size_t sharedMemSize = 2 * blockSize.x * blockSize.y * sizeof(int);

    float ms_shared = measure_cuda_time([&]() {
        cuda_matrix_multiply_shared<<<gridSize, blockSize, sharedMemSize>>>(devA, devB, devC, M, K, N);
    });

    cudaMemcpy(C_shared, devC, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    if(std::equal(C_global, C_global + M*N, C_shared)) printf("Matrices are equal\n");  

    printf("Kernel Global Memory Time: %.4f ms\n", ms_global);
    printf("Kernel Shared Memory Time: %.4f ms\n", ms_shared);

    cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C_global); cudaFreeHost(C_shared);
    cudaFree(devA); cudaFree(devB); cudaFree(devC);
}

int main(int argc, char** argv){
    int M = 0;
    int K = 0;
    int N = 0;
    bool serial = 0;
    std::srand(std::time(nullptr));
    if(argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        if (M <= 0 || K <= 0 || N <= 0){
            printf("Dimensions must be greater than 0\n");
            return 1;
        }
    }
    if (argc >= 5) serial = std::string(argv[4]) == "--serial";
    if (argc == 1) {
        M = 2;
        K = 2;
        N = 2;
    }
    else if(argc < 4){
        printf("Usage: %s <M> <K> <N> <--serial>", argv[0]);
        return 1;
    }
    matrix_multiply(M, K, N, serial);
}