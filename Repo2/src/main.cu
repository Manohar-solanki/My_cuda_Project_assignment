#include <cuda_runtime.h>
#include <cufft.h>
#include <sndfile.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

extern float computeAnomalyScore(cufftComplex* d_fft, int n);

void savePGM(const char* filename, const float* data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        unsigned char val = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, data[i] * 255.0f)));
        file.write(reinterpret_cast<char*>(&val), 1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5 || strcmp(argv[1], "--input") || strcmp(argv[3], "--output")) {
        std::cerr << "Usage: " << argv[0] << " --input <file.wav> --output <file.pgm>\n";
        return 1;
    }

    const char* inputPath = argv[2];
    const char* outputPath = argv[4];

    SF_INFO sfInfo;
    SNDFILE* infile = sf_open(inputPath, SFM_READ, &sfInfo);
    if (!infile) {
        std::cerr << "Error opening input file\n";
        return 1;
    }

    if (sfInfo.channels != 1) {
        std::cerr << "Only mono .wav supported\n";
        sf_close(infile);
        return 1;
    }

    const int n = sfInfo.frames;
    std::vector<float> h_signal(n);
    sf_readf_float(infile, h_signal.data(), n);
    sf_close(infile);

    // Pad to next power of 2 for FFT
    int fftN = 1;
    while (fftN < n) fftN <<= 1;

    float *d_signal = nullptr;
    cufftComplex *d_fft = nullptr;
    cudaMalloc(&d_signal, fftN * sizeof(float));
    cudaMalloc(&d_fft, fftN * sizeof(cufftComplex));
    cudaMemcpy(d_signal, h_signal.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_signal + n, 0, (fftN - n) * sizeof(float));

    cufftHandle plan;
    cufftPlan1d(&plan, fftN, CUFFT_R2C, 1);
    auto start = std::chrono::high_resolution_clock::now();
    cufftExecR2C(plan, d_signal, d_fft);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float fftTime = std::chrono::duration<float, std::milli>(end - start).count();

    float anomalyScore = computeAnomalyScore(d_fft, fftN / 2 + 1);

    // Magnitude for spectrogram (simple: full signal = one column)
    std::vector<float> h_magnitude(fftN / 2 + 1);
    cufftComplex *h_temp = new cufftComplex[fftN / 2 + 1];
    cudaMemcpy(h_temp, d_fft, (fftN / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < fftN / 2 + 1; ++i) {
        h_magnitude[i] = log10f(sqrtf(h_temp[i].x * h_temp[i].x + h_temp[i].y * h_temp[i].y) + 1e-6f);
    }
    delete[] h_temp;

    float minVal = *std::min_element(h_magnitude.begin(), h_magnitude.end());
    float maxVal = *std::max_element(h_magnitude.begin(), h_magnitude.end());
    for (auto& v : h_magnitude) v = (v - minVal) / (maxVal - minVal + 1e-9f);

    savePGM(outputPath, h_magnitude.data(), 1, h_magnitude.size());

    std::cout << "FFT size: " << fftN << "\n";
    std::cout << "GPU FFT time: " << fftTime << " ms\n";
    std::cout << "Anomaly score: " << anomalyScore << " (" << (anomalyScore > 0.7f ? "ABNORMAL" : "HEALTHY") << ")\n";

    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_fft);
    return 0;
}