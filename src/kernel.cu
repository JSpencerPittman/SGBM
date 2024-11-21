#include <stdio.h>

#include "kernel.cuh"

__global__ void hello_from_kernel() {
    printf("Hello from CUDA! Block: %d, Thread: %d\n", blockIdx.x, threadIdx.x);
}

void hello_world() {
    hello_from_kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}