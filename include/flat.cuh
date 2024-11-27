#ifndef FLAT_H_
#define FLAT_H_

struct TensorDims {
    TensorDims(size_t rows, size_t cols, size_t channels):
        rows(rows), cols(cols), channels(channels) {}

    size_t rows;
    size_t cols;
    size_t channels;
};

template<typename T>
struct FlatTensor {
    FlatTensor(TensorDims dims, T* data): dims(dims), data(data) {}

    T* rowPtr(size_t row) {
        return data + (row * dims.cols * dims.channels); 
    }

    T* colPtr(size_t row, size_t col) {
        return data + (((row * dims.cols) + col) * dims.channels); 
    }

    T* valuePtr(size_t row, size_t col, size_t channel) {
        return data + (((row * dims.cols) + col) * dims.channels + channel); 
    }

    T& value(size_t row, size_t col, size_t channel) {
        return data[((row * dims.cols) + col) * dims.channels + channel];
    }

    size_t bytes() {
        return dims.rows * dims.cols * dims.channels * sizeof(T);
    }
    
    TensorDims dims;
    T* data;
};

#endif