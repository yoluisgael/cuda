#include <stdio.h>
#include <cuda/cmath>
#include <ctime>
#include <chrono>
#include <algorithm>
#include "cuda_utils.hpp"

__global__ void cuda_matrix_multiply(const int* matrix_a, const int* matrix_b, int* matrix_c, const int M, const int K, const int N){
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
    int* C = nullptr;

    int* devA = nullptr;
    int* devB = nullptr;
    int* devC = nullptr;

    cudaMallocHost(&A, M*K*sizeof(int));
    cudaMallocHost(&B, K*N*sizeof(int));
    cudaMallocHost(&C, M*N*sizeof(int));

    init_array(A, M*K);
    init_array(B, K*N);

    cudaMalloc(&devA, M*K*sizeof(int));
    cudaMalloc(&devB, K*N*sizeof(int));
    cudaMalloc(&devC, M*N*sizeof(int));

    cudaMemcpy(devA, A, M*K*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, K*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(devC, 0, M*N*sizeof(int));

    dim3 blockSize(32,32);
    dim3 gridSize((N + blockSize.x -1) / blockSize.x,
                  (M + blockSize.y -1) / blockSize.y);  

    float ms = measure_cuda_time([&]() {
        cuda_matrix_multiply<<<gridSize, blockSize>>>(devA, devB, devC, M, K, N);
    });

    cudaMemcpy(C, devC, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    print_matrix_10x10(A, M, K, "A");
    print_matrix_10x10(B, K, N, "B");
    print_matrix_10x10(C, M, N, "C");    

    if(serial){
        int* C_serial = nullptr;
        cudaMallocHost(&C_serial, M*N*sizeof(int));
        memset(C_serial, 0, M*N*sizeof(int));

        auto start = std::chrono::high_resolution_clock::now();

        serial_matrix_multiply(A, B, C_serial, M, K, N);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        print_matrix_10x10(C_serial, M, N, "C serial");
        if(std::equal(C, C + M*N, C_serial)) printf("Matrices are equal\n");         
        printf("CPU Time: %.4f ms\n", duration.count());

        cudaFreeHost(C_serial);
    }

    

    printf("Kernel Time: %.4f ms\n", ms);

    cudaFreeHost(A); cudaFreeHost(B); cudaFreeHost(C);
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