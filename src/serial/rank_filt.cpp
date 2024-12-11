#include "serial/sgbm.h"

TensorCoord clampImageCoordRankFilt(Tensor<Byte> &dispMap, int row, int col)
{
    return {
        static_cast<size_t>(std::max(std::min(row, static_cast<int>(dispMap.dims.rows) - 1), 0)),
        static_cast<size_t>(std::max(std::min(col, static_cast<int>(dispMap.dims.cols) - 1), 0)),
    };
}

void selectionSort(Byte *arr, size_t len)
{
    for (size_t idxInsert = 0; idxInsert < len - 1; ++idxInsert)
    {
        size_t minIdx = idxInsert;
        for (size_t idxSearch = idxInsert + 1; idxSearch < len; ++idxSearch)
            if (arr[idxSearch] < arr[minIdx])
                minIdx = idxSearch;

        Byte temp = arr[idxInsert];
        arr[idxInsert] = arr[minIdx];
        arr[minIdx] = temp;
    }
}

void medianRankFilterAtPixel(Tensor<Byte> &dispMap,
                             Tensor<Byte> &blurDispMap,
                             TensorCoord coordDispMap,
                             Byte *kernelCache)
{
    for (int diffY = -MORPH_RADIUS; diffY <= MORPH_RADIUS; ++diffY)
        for (int diffX = -MORPH_RADIUS; diffX <= MORPH_RADIUS; ++diffX)
        {
            TensorCoord loc = clampImageCoordRankFilt(dispMap,
                                                      static_cast<int>(coordDispMap.row) + diffY,
                                                      static_cast<int>(coordDispMap.col) + diffX);
            kernelCache[(diffY + MORPH_RADIUS) * MORPH_DIAMETER + (diffX + MORPH_RADIUS)] =
                dispMap(loc);
        }

    selectionSort(kernelCache, MORPH_DIAMETER * MORPH_DIAMETER);
    size_t middleIdx = ((MORPH_DIAMETER * MORPH_DIAMETER) - 1) / 2;

    Byte median = kernelCache[middleIdx];
    blurDispMap(coordDispMap) = median;
}

void medianRankFilterKernel(Tensor<Byte> &dispMap, Tensor<Byte> &blurDispMap)
{
    Byte kernelCache[MORPH_DIAMETER * MORPH_DIAMETER];

    for (size_t r = 0; r < dispMap.dims.rows; ++r)
    {
        for (size_t c = 0; c < dispMap.dims.cols; ++c)
        {
            TensorCoord coordDispMap(r, c);
            medianRankFilterAtPixel(dispMap, blurDispMap, coordDispMap, kernelCache);
        }
    }
}

void minMaxFilterAtPixel(Tensor<Byte> &dispMap,
                         Tensor<Byte> &blurDispMap,
                         TensorCoord coordDispMap,
                         bool isMaxOperation)
{
    Byte value = blurDispMap(coordDispMap);
    for (int diffY = -MORPH_RADIUS; diffY <= MORPH_RADIUS; ++diffY)
        for (int diffX = -MORPH_RADIUS; diffX <= MORPH_RADIUS; ++diffX)
        {
            TensorCoord loc = clampImageCoordRankFilt(dispMap,
                                                      static_cast<int>(coordDispMap.row) + diffY,
                                                      static_cast<int>(coordDispMap.col) + diffX);
            if (isMaxOperation)
                value = std::max(value, dispMap(loc));
            else
                value = std::min(value, dispMap(loc));
        }

    blurDispMap(coordDispMap) = value;
}

void minMaxRankFilterKernel(Tensor<Byte> &dispMap, Tensor<Byte> &blurDispMap, bool isMaxOperation)
{
    for (size_t r = 0; r < dispMap.dims.rows; ++r)
    {
        for (size_t c = 0; c < dispMap.dims.cols; ++c)
        {
            TensorCoord coordDispMap(r, c);
            minMaxFilterAtPixel(dispMap, blurDispMap, coordDispMap, isMaxOperation);
        }
    }
}

Tensor<Byte> allocateBlurDispMap(Tensor<Byte> &image)
{
    return {image.dims};
}

Tensor<Byte> rankFilter(Tensor<Byte> &dispMap, RankOperation operation)
{
    Tensor<Byte> blurDispMap = allocateBlurDispMap(dispMap);

    if (operation == RankOperation::Median)
        medianRankFilterKernel(dispMap, blurDispMap);
    else
        minMaxRankFilterKernel(dispMap, blurDispMap, operation == RankOperation::Maximum);

    return blurDispMap;
}
