#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

using namespace nvcuda;

__global__ void wmma_gemm_16x16x16(const half* A, const half* B, float* C) {
    // 最小示例：1 个 warp 计算 1 个 16x16 的输出 tile。
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main() {
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;
    constexpr int A_ELEMS = M * K;
    constexpr int B_ELEMS = K * N;
    constexpr int C_ELEMS = M * N;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::cout << "This WMMA demo requires SM >= 70 (Tensor Cores)." << std::endl;
        return 0;
    }

    std::vector<half> hA(A_ELEMS);
    std::vector<half> hB(B_ELEMS);
    std::vector<float> hC(C_ELEMS, 0.0f);
    std::vector<float> hRef(C_ELEMS, 0.0f);

    // 构造小范围数据，方便观察与验证。
    for (int i = 0; i < A_ELEMS; ++i) {
        float v = static_cast<float>((i % 7) - 3) * 0.25f;
        hA[i] = __float2half(v);
    }
    for (int i = 0; i < B_ELEMS; ++i) {
        float v = static_cast<float>((i % 5) - 2) * 0.5f;
        hB[i] = __float2half(v);
    }

    half* dA = nullptr;
    half* dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, A_ELEMS * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, B_ELEMS * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, C_ELEMS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), A_ELEMS * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), B_ELEMS * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, C_ELEMS * sizeof(float)));

    // 1 个 block, 1 个 warp（32 线程）
    wmma_gemm_16x16x16<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, C_ELEMS * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 参考结果：FP16 输入转 FP32 再做累加
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = __half2float(hA[m * K + k]);
                float b = __half2float(hB[k * N + n]);
                sum += a * b;
            }
            hRef[m * N + n] = sum;
        }
    }

    float max_abs_err = 0.0f;
    for (int i = 0; i < C_ELEMS; ++i) {
        max_abs_err = std::max(max_abs_err, std::fabs(hC[i] - hRef[i]));
    }

    std::cout << "WMMA 16x16x16 GEMM (FP16 input, FP32 accumulate)\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "max_abs_err = " << std::setprecision(6) << max_abs_err << "\n";
    std::cout << "check: " << (max_abs_err < 1e-2f ? "PASS" : "FAILED") << "\n";
    std::cout << "C[0..7]: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << hC[i] << (i + 1 == 8 ? '\n' : ' ');
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
