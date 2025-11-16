#include <cuda_runtime.h>
#include <cufft.h>
#include <cooperative_groups.h>

__global__ void computeSpectralCentroid(cufftComplex* d_fft, float* d_centroid, int n) {
    float sum = 0.0f, total = 0.0f;
    for (int i = 1; i < n; ++i) {
        float mag = sqrtf(d_fft[i].x * d_fft[i].x + d_fft[i].y * d_fft[i].y);
        sum += i * mag;
        total += mag;
    }
    d_centroid[0] = (total > 1e-6f) ? sum / total : 0.0f;
}

float computeAnomalyScore(cufftComplex* d_fft, int n) {
    float *d_centroid = nullptr, h_centroid;
    cudaMalloc(&d_centroid, sizeof(float));
    computeSpectralCentroid<<<1, 1>>>(d_fft, d_centroid, n);
    cudaMemcpy(&h_centroid, d_centroid, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_centroid);

    // Normalize: healthy = low centroid (~200-800 Hz), abnormal = high (>1000 Hz)
    // Assume sampling rate = 22050 Hz, bin width = 22050 / (2*n)
    float binToHz = 22050.0f / (2 * (n - 1));
    float centroidHz = h_centroid * binToHz;
    float score = fminf(1.0f, centroidHz / 1500.0f); // Scale: 1500 Hz = max score
    return score;
}