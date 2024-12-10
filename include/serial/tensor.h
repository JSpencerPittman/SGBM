#ifndef TENSOR_SERIAL_H_
#define TENSOR_SERIAL_H_

#include <cstdint>
#include <memory>

struct TensorCoord
{
    TensorCoord(size_t row, size_t col) : row(row), col(col), channel(0) {}
    TensorCoord(size_t row, size_t col, size_t channel) : row(row), col(col), channel(channel) {}

    size_t row;
    size_t col;
    size_t channel;
};

struct TensorDims
{
    TensorDims(size_t rows, size_t cols, size_t channels) : rows(rows), cols(cols), channels(channels) {}

    size_t rows;
    size_t cols;
    size_t channels;
};

template <typename T>
struct Tensor
{
    Tensor(TensorDims dims) : dims(dims), data(nullptr)
    {
        data = new T[dims.rows * dims.cols * dims.channels];
    }

    Tensor(TensorDims dims, T *data) : dims(dims), data(data) {}

    void free()
    {
        delete[] data;
    }

    T *rowPtr(TensorCoord &coord)
    {
        return rowPtr(coord.row);
    }

    T *rowPtr(size_t row)
    {
        return data + (row * dims.cols * dims.channels);
    }

    T *colPtr(TensorCoord &coord)
    {
        return colPtr(coord.row, coord.col);
    }

    T *colPtr(size_t row, size_t col)
    {
        return data + (((row * dims.cols) + col) * dims.channels);
    }

    T *valuePtr(TensorCoord &coord)
    {
        return valuePtr(coord.row, coord.col, coord.channel);
    }

    T *valuePtr(size_t row, size_t col, size_t channel)
    {
        return data + (((row * dims.cols) + col) * dims.channels + channel);
    }

    T &operator()(TensorCoord &coord)
    {
        return value(coord);
    }

    T &operator()(size_t row, size_t col, size_t channel = 0)
    {
        return value(row, col, channel);
    }

    T &value(TensorCoord &coord)
    {
        return value(coord.row, coord.col, coord.channel);
    }

    T &value(size_t row, size_t col, size_t channel = 0)
    {
        return data[((row * dims.cols) + col) * dims.channels + channel];
    }

    size_t bytes()
    {
        return dims.rows * dims.cols * dims.channels * sizeof(T);
    }

    TensorDims dims;
    T *data;
};

#endif