# Image Processing and Analysis using NVIDIA NPP with CUDA

## Overview

This project demonstrates an advanced image processing pipeline leveraging the **NVIDIA Performance Primitives (NPP)** library to perform restoration and feature extraction of medical imagery, specifically **64-bit medical TIFF** CT scans, using GPU acceleration. The project implements **Richardson-Lucy Deconvolution** to deblur images, incorporates **Tikhonov Regularization** for noise suppression, and utilizes **Medical Windowing** for high-precision visualization. By offloading these intensive mathematical operations to the GPU, the system achieves high-throughput processing suitable for enterprise-scale medical data analysis.

---

## Code Organization

* **`bin/`**: Holds all binary/executable code, such as the compiled `final_project.exe`.
* **`data/`**: Holds example data, including input medical imagery like `CT_scan.tif` and the resulting processed output files.
* **`lib/`**: Contains libraries not installed via the OS-specific package manager, such as `stb_image_write.h` and links to `libtiff`.
* **`src/`**: The source code, primarily `main.cu`, is placed here in a hierarchical fashion.
* **`README.md`**: Holds the description of the project and its purpose to help with the decision to clone the repository.
* **`INSTALL`**: Holds human-readable instructions for installing the code so that it can be executed across different operating systems.
* **`Makefile`**: Rudimentary script for building the project's code in an automatic fashion using `nvcc`.
* **`run.sh`**: Optional script used to run the executable code with or without command-line arguments.

---

## Key Concepts

* **Performance Strategies**: Utilizing GPU hardware features like global and register memory.
* **Image Processing**: Handling high-bit depth imagery and medical data reconstruction.
* **NPP Library**: Leveraging NVIDIA's optimized primitives for accelerated processing.
* **Richardson-Lucy Deconvolution**: An iterative Bayesian method for image restoration.
* **CUDA Streams**: Enabling asynchronous concurrent execution and batch processing.



---

## Supported SM Architectures

[SM 3.5](https://developer.nvidia.com/cuda-gpus) | [SM 5.0](https://developer.nvidia.com/cuda-gpus) | [SM 6.0](https://developer.nvidia.com/cuda-gpus) | [SM 7.0](https://developer.nvidia.com/cuda-gpus) | [SM 7.5](https://developer.nvidia.com/cuda-gpus) | [SM 8.0](https://developer.nvidia.com/cuda-gpus) | [SM 8.6](https://developer.nvidia.com/cuda-gpus).

## Supported OSes / CPU Architectures

* **OS**: Linux, Windows.
* **CPU**: x86_64, ppc64le, armv7l.

---

## Function Descriptions

### 1. GPU Kernels (Device Code)
* **`ComplexMultiply`**: Performs element-wise multiplication of complex matrices in the frequency domain, core to the convolution process.
* **`ComplexDivide`**: Handles division of the blurred image by the current estimate, using an epsilon to prevent division-by-zero errors.
* **`RegularizedUpdate`**: Implements **Tikhonov Regularization** to update the estimate while suppressing high-frequency noise artifacts.
* **`NormalizeAndScale`**: Adjusts unnormalized `CUFFT_INVERSE` output by $1/N$ to keep intensities within valid mathematical bounds.

### 2. Host Logic (C++ Code)
* **`ImageTask (Class)`**: Employs **RAII** and **CUDA Streams** to manage GPU resources and enable concurrent file processing.
* **`TIFFReadScanline`**: A high-precision I/O function that reads **64-bit samples** from TIFF files and converts them for GPU processing.
* **`log1p Pre-Normalization`**: Compresses the massive dynamic range of Hounsfield Units before the iterative loop to prevent numerical overflow.
* **`Percentile Windowing`**: Maps the 2nd to 98th percentile of intensities to 0-255, revealing soft-tissue details while ignoring outliers.

---

## Statistical Inferences

The program generates logs to verify performance and accuracy:

* **Execution Time (s)**: Measured using `cudaEvent_t`. Asynchronous streams allow overlapping data transfers and computation to maximize hardware utilization.
* **Mean Difference Percentage**: The primary grading metric comparing recovered estimates to initial blurred inputs.
    * **High Percentage (90%+)**: Indicates significant feature recovery and successful medical detail reconstruction.
    * **Low Percentage**: Suggests the regularization parameter ($\lambda$) is too high or the PSF does not match the actual blur.



---

## Build and Run

Compile using: nvcc -o rl_deblur main.cu -lcufft -ltiff -std=c++17 

Run using: ./rl_deblur