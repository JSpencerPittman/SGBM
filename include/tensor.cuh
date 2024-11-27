#ifndef FLAT_H_
#define FLAT_H_

#include<cuda_runtime.h>

struct TensorCoord {
    __host__ __device__ TensorCoord(size_t row, size_t col):
        row(row), col(col), channel(0) {}
    __host__ __device__ TensorCoord(size_t row, size_t col, size_t channel):
        row(row), col(col), channel(channel) {}

    size_t row;
    size_t col;
    size_t channel;
};

struct TensorDims {
    __host__ __device__ TensorDims(size_t rows, size_t cols, size_t channels):
        rows(rows), cols(cols), channels(channels) {}

    size_t rows;
    size_t cols;
    size_t channels;
};

template<typename T>
struct Tensor {
    __host__ __device__ Tensor(TensorDims dims, bool gpu = false): 
        dims(dims), data(nullptr), gpu(gpu) {
            if(gpu) cudaMalloc(&data, bytes());
            else data = new T[dims.rows * dims.cols * dims.channels];
        }

    __host__ __device__ Tensor(TensorDims dims, T* data, bool gpu = false): 
        dims(dims), data(data), gpu(gpu) {} 

    Tensor<T> copyToHost() {
        Tensor<T> tensorHost(dims, false);
        cudaMemcpy(tensorHost.data, data, bytes(), cudaMemcpyDeviceToHost);
        return tensorHost;
    }

    Tensor<T> copyToDevice() {
        Tensor<T> tensorDev(dims, true);
        cudaMemcpy(tensorDev.data, data, bytes(), cudaMemcpyHostToDevice);
        return tensorDev;
    }

    __host__ __device__ void free() {
        if(gpu) cudaFree(data);
        else delete [] data;
    }

    __host__ __device__ T* rowPtr(TensorCoord& coord) {
        return rowPtr(coord.row);
    }

    __host__ __device__ T* rowPtr(size_t row) {
        return data + (row * dims.cols * dims.channels); 
    }

    __host__ __device__ T* colPtr(TensorCoord& coord) {
        return colPtr(coord.row, coord.col); 
    }

    __host__ __device__ T* colPtr(size_t row, size_t col) {
        return data + (((row * dims.cols) + col) * dims.channels); 
    }

    __host__ __device__ T* valuePtr(TensorCoord& coord) {
        return valuePtr(coord.row, coord.col, coord.channel); 
    }

    __host__ __device__ T* valuePtr(size_t row, size_t col, size_t channel) {
        return data + (((row * dims.cols) + col) * dims.channels + channel); 
    }

    __host__ __device__ T& operator()(TensorCoord& coord) {
        return value(coord);
    }

    __host__ __device__ T& operator()(size_t row, size_t col, size_t channel = 0) {
        return value(row, col, channel);
    }

    __host__ __device__ T& value(TensorCoord& coord) {
        return value(coord.row, coord.col, coord.channel); 
    }

    __host__ __device__ T& value(size_t row, size_t col, size_t channel = 0) {
        return data[((row * dims.cols) + col) * dims.channels + channel];
    }

    __host__ __device__ size_t bytes() {
        return dims.rows * dims.cols * dims.channels * sizeof(T);
    }
    
    bool gpu;
    TensorDims dims;
    T* data;
};

#endif