#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

// 简单计算核函数：给数组每个元素做多次浮点运算，拉长执行时间，便于观察stream效果。
__global__ void transform_kernel(const float* in, float* out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        for (int i = 0; i < iters; ++i) {
            x = x * 1.000001f + 0.000001f;
        }
        out[idx] = x;
    }
}

static float run_single_default_stream(const float* h_in, float* h_out, int n, int iters) {
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    transform_kernel<<<blocks, threads>>>(d_in, d_out, n, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ms;
}

static float run_two_streams(const float* h_in, float* h_out, int n, int iters) {
    const int chunks = 2;
    const int chunk_n = n / chunks;
    const size_t chunk_bytes = static_cast<size_t>(chunk_n) * sizeof(float);

    cudaStream_t streams[chunks];
    float* d_in[chunks] = {nullptr, nullptr};
    float* d_out[chunks] = {nullptr, nullptr};

    for (int i = 0; i < chunks; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_in[i], chunk_bytes));
        CUDA_CHECK(cudaMalloc(&d_out[i], chunk_bytes));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int threads = 256;
    int blocks = (chunk_n + threads - 1) / threads;
    for (int i = 0; i < chunks; ++i) {
        const float* h_in_chunk = h_in + static_cast<size_t>(i) * chunk_n;
        float* h_out_chunk = h_out + static_cast<size_t>(i) * chunk_n;
        CUDA_CHECK(cudaMemcpyAsync(
            d_in[i], h_in_chunk, chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        transform_kernel<<<blocks, threads, 0, streams[i]>>>(d_in[i], d_out[i], chunk_n, iters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(
            h_out_chunk, d_out[i], chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int i = 0; i < chunks; ++i) {
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    return ms;
}

int main() {
    const int n = 1 << 22;  // 约 4M 元素
    const int iters = 300;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // 使用页锁定内存，cudaMemcpyAsync 才能真正异步。
    float *h_in = nullptr, *h_out_default = nullptr, *h_out_stream = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out_default, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out_stream, bytes));

    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i % 1024) * 0.001f;
    }

    float default_ms = run_single_default_stream(h_in, h_out_default, n, iters);
    float stream_ms = run_two_streams(h_in, h_out_stream, n, iters);

    int mismatch = 0;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(h_out_default[i] - h_out_stream[i]);
        if (diff > 1e-4f) {
            mismatch = 1;
            break;
        }
    }

    std::cout << "=== CUDA Stream Demo ===" << std::endl;
    std::cout << "Data size: " << n << " elements" << std::endl;
    std::cout << "Default stream (serial copy+compute): " << default_ms << " ms" << std::endl;
    std::cout << "Two streams (overlap copy+compute):   " << stream_ms << " ms" << std::endl;
    if (stream_ms > 0.0f) {
        std::cout << "Speedup: " << (default_ms / stream_ms) << "x" << std::endl;
    }
    std::cout << "Result check: " << (mismatch ? "FAILED" : "PASS") << std::endl;

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out_default));
    CUDA_CHECK(cudaFreeHost(h_out_stream));
    return mismatch ? EXIT_FAILURE : EXIT_SUCCESS;
}
