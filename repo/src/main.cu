#include <cuda_runtime.h>
#include <cufft.h>
#include <sndfile.h>
#include <png.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <dirent.h>
#include <sys/time.h>

// Helpers
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Save 1D magnitude as grayscale PNG
void save_spectrogram(const float* mag, int width, int height, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    std::vector<png_bytep> rows(height);
    std::vector<std::vector<png_byte>> data(height, std::vector<png_byte>(width));
    float max_val = 0;
    for (int i = 0; i < width * height; ++i)
        if (mag[i] > max_val) max_val = mag[i];

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            data[y][x] = (png_byte)(255 * mag[y * width + x] / (max_val + 1e-9));

    for (int y = 0; y < height; ++y) rows[y] = data[y].data();
    png_write_image(png, rows.data());
    png_write_end(png, NULL);

    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>\n";
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    DIR* dir = opendir(input_dir.c_str());
    if (!dir) {
        std::cerr << "Cannot open input directory\n";
        return 1;
    }

    double total_start = get_time();
    int file_count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (!ends_with(entry->d_name, ".wav")) continue;

        std::string filepath = input_dir + "/" + entry->d_name;
        SF_INFO sf_info;
        SNDFILE* sndfile = sf_open(filepath.c_str(), SFM_READ, &sf_info);
        if (!sndfile) continue;

        if (sf_info.channels != 1) {
            sf_close(sndfile);
            continue; // Skip stereo
        }

        std::vector<float> host_input(sf_info.frames);
        sf_read_float(sndfile, host_input.data(), sf_info.frames);
        sf_close(sndfile);

        // GPU setup
        float *d_input, *d_mag;
        cufftComplex *d_output;
        size_t input_bytes = sf_info.frames * sizeof(float);
        size_t output_bytes = (sf_info.frames / 2 + 1) * sizeof(cufftComplex);
        cudaMalloc(&d_input, input_bytes);
        cudaMalloc(&d_output, output_bytes);
        cudaMalloc(&d_mag, sf_info.frames * sizeof(float));

        cudaMemcpy(d_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice);

        // cuFFT plan
        cufftHandle plan;
        cufftPlan1d(&plan, sf_info.frames, CUFFT_R2C, 1);
        cufftExecR2C(plan, d_input, d_output);

        // Compute magnitude
        auto mag_kernel = [] __device__ (cufftComplex* out, float* mag, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N) {
                float real = out[i].x;
                float imag = out[i].y;
                mag[i] = sqrtf(real * real + imag * imag);
            }
        };
        int threads = 256;
        int blocks = (sf_info.frames + threads - 1) / threads;
        mag_kernel<<<blocks, threads>>>(d_output, d_mag, sf_info.frames);
        cudaDeviceSynchronize();

        // Copy back
        std::vector<float> host_mag(sf_info.frames);
        cudaMemcpy(host_mag.data(), d_mag, sf_info.frames * sizeof(float), cudaMemcpyDeviceToHost);

        // Save spectrogram (1D → 2D reshape for PNG)
        int width = 512;
        int height = (sf_info.frames + width - 1) / width;
        save_spectrogram(host_mag.data(), width, height,
                         (output_dir + "/" + std::string(entry->d_name) + ".png").c_str());

        // Cleanup
        cufftDestroy(plan);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_mag);
        file_count++;
        if (file_count >= 3) break; // For demo; remove for full run
    }
    closedir(dir);

    double total_time = get_time() - total_start;
    std::ofstream log(output_dir + "/execution.log");
    log << "Processed " << file_count << " files in " << total_time << " seconds\n";
    log.close();

    return 0;
}