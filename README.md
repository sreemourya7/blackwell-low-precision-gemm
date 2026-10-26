# Blackwell Low-Precision GEMM

**FP4/FP8 Block-Scaled Matrix Multiplication on NVIDIA RTX 5090 (SM120)**

CUDA C++ · PTX · NVFP4 · MXFP8 · Tensor Cores · Nsight Compute · CUTLASS

## Project Overview

This project explores the design and optimization of low-precision, block-scaled matrix multiplication kernels on NVIDIA Blackwell GPUs, with a focus on native Tensor Core execution and GPU performance engineering.

Modern AI workloads rely heavily on matrix multiplication, particularly in transformer attention, projection, and feed-forward layers. Lower-precision formats such as FP4 and FP8 reduce memory footprint and data movement while enabling higher Tensor Core throughput.

However, aggressively reducing precision introduces numerical challenges. Block scaling addresses these limitations by assigning scale factors to small groups of values, helping preserve local dynamic range while retaining the computational benefits of low-precision arithmetic.

The project investigates the complete path from quantization and scale-factor representation to native Blackwell matrix instructions, memory-layout optimization, scheduling, and hardware-level profiling.

**The broader goal is to understand how to develop high-performance custom GEMM kernels for AI workloads**, including transformer inference, quantized linear layers, and specialized matrix shapes where generic library kernels may not provide optimal performance.

## Project Status

Development is organized into incremental stages covering quantization, correctness validation, CUDA kernel implementation, native Tensor Core execution, optimization, benchmarking, and numerical accuracy.

Completed development work, source code, profiling observations, and benchmark results are being consolidated and transferred from the engineering worklog into this repository.

Detailed implementation notes and verified performance results will be published as the corresponding artifacts are integrated.