#include <iostream>
#include <cublas_v2.h>

int main() {
    const int N = 3;
    const int M = 2;
    const int K = 3;

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, N * K * sizeof(float));
    cudaMallocHost(&h_B, K * M * sizeof(float));
    cudaMallocHost(&h_C, N * M * sizeof(float));

    // Initialize matrices h_A and h_B
    for (int i = 0; i < N * K; i++) {
        h_A[i] = i;
    }

    for (int i = 0; i < K * M; i++) {
        h_B[i] = i;
    }

    // Allocate device memory and copy data to device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * K * sizeof(float));
    cudaMalloc(&d_B, K * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));

    cublasSetMatrix(N, K, sizeof(float), h_A, N, d_A, N);
    cublasSetMatrix(K, M, sizeof(float), h_B, K, d_B, K);

    // Perform matrix multiplication
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_A, N, d_B, K, &beta, d_C, N);

    // Copy the result back to host
    cublasGetMatrix(N, M, sizeof(float), d_C, N, h_C, N);

    // Display the result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << h_C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    cublasDestroy(handle);

    return 0;
}