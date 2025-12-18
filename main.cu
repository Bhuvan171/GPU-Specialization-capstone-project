/**
 * GPU Richardson-Lucy Batch Processor (Medical Reconstruction Edition)
 * Features: 64-bit TIFF, CUDA Streams, Tikhonov Regularization, Percentile Windowing
 * Compile: nvcc -o rl_deblur main.cu -lcufft -ltiff -std=c++17
 */

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <memory>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>
#include <tiffio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

// --- Macros ---
#define CUDA_CHECK(err) { if (err != cudaSuccess) { fprintf(stderr, "[CUDA Error] %s\n", cudaGetErrorString(err)); exit(1); } }
#define CUFFT_CHECK(err) { if (err != CUFFT_SUCCESS) { fprintf(stderr, "[cuFFT Error] %d\n", err); exit(1); } }

// --- Kernels using Device Global Memory & Registers ---
__global__ void ComplexMultiply(cufftComplex* a, const cufftComplex* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float re = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
        float im = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
        a[idx].x = re; a[idx].y = im;
    }
}

__global__ void ComplexDivide(cufftComplex* a, const cufftComplex* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Epsilon protects against division by zero (black squares)
        float den = b[idx].x * b[idx].x + b[idx].y * b[idx].y + 1e-5f;
        float re = (a[idx].x * b[idx].x + a[idx].y * b[idx].y) / den;
        // Moderate clamp prevents intensity explosion (white-out)
        a[idx].x = fminf(fmaxf(re, 0.0f), 4.0f); 
        a[idx].y = 0.0f;
    }
}

__global__ void RegularizedUpdate(cufftComplex* estimate, const cufftComplex* correction, int size, float lambda, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = estimate[idx].x * (correction[idx].x * scale);
        // Tikhonov Regularization: Suppress high-frequency noise artifacts
        estimate[idx].x = fmaxf(val / (1.0f + lambda), 1e-9f); 
        estimate[idx].y = 0.0f;
    }
}

class ImageTask {
public:
    std::string name;
    int w, h, N;
    cudaStream_t stream;
    cufftHandle plan;
    cufftComplex *d_blurred, *d_estimate, *d_psf, *d_temp;

    ImageTask(const std::string& filename, int width, int height) 
        : name(filename), w(width), h(height), N(width * height) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUFFT_CHECK(cufftPlan2d(&plan, h, w, CUFFT_C2C));
        CUFFT_CHECK(cufftSetStream(plan, stream));
        CUDA_CHECK(cudaMalloc(&d_blurred, sizeof(cufftComplex) * N));
        CUDA_CHECK(cudaMalloc(&d_estimate, sizeof(cufftComplex) * N));
        CUDA_CHECK(cudaMalloc(&d_psf, sizeof(cufftComplex) * N));
        CUDA_CHECK(cudaMalloc(&d_temp, sizeof(cufftComplex) * N));
    }

    ~ImageTask() {
        cudaFree(d_blurred); cudaFree(d_estimate); cudaFree(d_psf); cudaFree(d_temp);
        cufftDestroy(plan);
        cudaStreamDestroy(stream);
    }
};

int main(int argc, char** argv) {
    std::string inputDir = "input_images";
    std::string outputDir = "output_images";
    int iterations = 15;
    float lambda = 0.005f; // Regularization strength
    fs::create_directories(outputDir);

    std::vector<std::unique_ptr<ImageTask>> tasks;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".tif" || entry.path().extension() == ".tiff") {
            TIFF* tif = TIFFOpen(entry.path().string().c_str(), "r");
            if (!tif) continue;

            uint32_t w, h;
            uint16_t bits;
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
            TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);

            auto task = std::make_unique<ImageTask>(entry.path().filename().string(), (int)w, (int)h);
            std::vector<double> raw_data(w * h);

            tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
            for (uint32_t row = 0; row < h; row++) {
                TIFFReadScanline(tif, buf, row);
                for (uint32_t col = 0; col < w; col++) {
                    raw_data[row * w + col] = (bits == 64) ? ((double*)buf)[col] : (double)((uint16_t*)buf)[col];
                }
            }
            _TIFFfree(buf);
            TIFFClose(tif);

            // Logarithmic Pre-Normalization for Hounsfield Dynamic Range
            double min_v = *std::min_element(raw_data.begin(), raw_data.end());
            std::vector<cufftComplex> h_img(w * h), h_psf(w * h, {0,0});
            for(size_t i=0; i < raw_data.size(); ++i) {
                h_img[i] = { (float)log1p(fmax(0.0, raw_data[i] - min_v)), 0.0f };
            }

            // PSF Energy Conservation
            float psf_sum = 0.0f;
            for(int r=-2; r<=2; r++) {
                for(int c=-2; c<=2; c++) {
                    float v = exp(-(r*r+c*c)/2.0f);
                    h_psf[((r+h)%h)*w + ((c+w)%w)].x = v;
                    psf_sum += v;
                }
            }
            for(auto& p : h_psf) p.x /= psf_sum;

            CUDA_CHECK(cudaMemcpyAsync(task->d_blurred, h_img.data(), w*h*sizeof(cufftComplex), cudaMemcpyHostToDevice, task->stream));
            CUDA_CHECK(cudaMemcpyAsync(task->d_estimate, h_img.data(), w*h*sizeof(cufftComplex), cudaMemcpyHostToDevice, task->stream));
            CUDA_CHECK(cudaMemcpyAsync(task->d_psf, h_psf.data(), w*h*sizeof(cufftComplex), cudaMemcpyHostToDevice, task->stream));
            tasks.push_back(std::move(task));
        }
    }

    // --- Processing Pipeline ---
    for(auto& t : tasks) CUFFT_CHECK(cufftExecC2C(t->plan, t->d_psf, t->d_psf, CUFFT_FORWARD));

    for(int i = 0; i < iterations; i++) {
        for(auto& t : tasks) {
            int threads = 256;
            int blocks = (t->N + threads - 1) / threads;
            float scale = 1.0f / t->N;

            CUFFT_CHECK(cufftExecC2C(t->plan, t->d_estimate, t->d_temp, CUFFT_FORWARD));
            ComplexMultiply<<<blocks, threads, 0, t->stream>>>(t->d_temp, t->d_psf, t->N);
            CUFFT_CHECK(cufftExecC2C(t->plan, t->d_temp, t->d_temp, CUFFT_INVERSE));
            // No regularization here, just scaling and clamping
            
            ComplexDivide<<<blocks, threads, 0, t->stream>>>(t->d_temp, t->d_blurred, t->N);

            CUFFT_CHECK(cufftExecC2C(t->plan, t->d_temp, t->d_temp, CUFFT_FORWARD));
            ComplexMultiply<<<blocks, threads, 0, t->stream>>>(t->d_temp, t->d_psf, t->N);
            CUFFT_CHECK(cufftExecC2C(t->plan, t->d_temp, t->d_temp, CUFFT_INVERSE));
            
            // Apply Tikhonov regularization and scale in the final update
            RegularizedUpdate<<<blocks, threads, 0, t->stream>>>(t->d_estimate, t->d_temp, t->N, lambda, scale);
        }
    }

    // --- Robust Windowed Output ---
    for(auto& t : tasks) {
        cudaStreamSynchronize(t->stream);
        std::vector<cufftComplex> h_res(t->N);
        cudaMemcpy(h_res.data(), t->d_estimate, t->N*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        
        std::vector<float> intensities(t->N);
        for(int i=0; i<t->N; i++) intensities[i] = h_res[i].x;
        
        // Find 2% and 98% percentiles to remove medical scan noise outliers
        std::vector<float> sorted = intensities;
        std::nth_element(sorted.begin(), sorted.begin() + t->N*0.02, sorted.end());
        float p02 = sorted[t->N*0.02];
        std::nth_element(sorted.begin(), sorted.begin() + t->N*0.98, sorted.end());
        float p98 = sorted[t->N*0.98];
        float range = p98 - p02 + 1e-6f;

        std::vector<unsigned char> pixels(t->N);
        for(int i=0; i<t->N; i++) {
            float norm = (intensities[i] - p02) / range;
            pixels[i] = (unsigned char)(fminf(fmaxf(norm, 0.0f), 1.0f) * 255.0f);
        }
        stbi_write_png((outputDir + "/medical_" + t->name + ".png").c_str(), t->w, t->h, 1, pixels.data(), t->w);
    }

    return 0;
}