#include "serial/sgbm.h"

#include <algorithm>

struct ImgCoord
{
    ImgCoord(int x, int y) : x(x), y(y) {}
    int x, y;
};

namespace Direction
{
    enum Direction
    {
        LeftToRight,
        TopLeftToBotRight,
        TopToBot,
        TopRightToBotLeft,
        RightToLeft,
        BotRightToTopLeft,
        BotToTop,
        BotLeftToTopRight
    };

    ImgCoord start(Direction direction, size_t idx, size_t width, size_t height)
    {
        switch (direction)
        {
        case LeftToRight:
            return {0, static_cast<int>(idx)};
        case TopLeftToBotRight:
            if (idx < height)
                return {0, static_cast<int>(idx)};
            else
                return {static_cast<int>(idx - height + 1), 0};
        case TopToBot:
            return {static_cast<int>(idx), 0};
        case TopRightToBotLeft:
            if (idx < width)
                return {static_cast<int>(idx), 0};
            else
                return {static_cast<int>(width - 1), static_cast<int>(idx - width + 1)};
        case RightToLeft:
            return {static_cast<int>(width - 1), static_cast<int>(idx)};
        case BotRightToTopLeft:
            if (idx < width)
                return {static_cast<int>(idx), static_cast<int>(height - 1)};
            else
                return {static_cast<int>(width - 1), static_cast<int>(idx - width)};
        case BotToTop:
            return {static_cast<int>(idx), static_cast<int>(height - 1)};
        case BotLeftToTopRight:
            if (idx < width)
                return {static_cast<int>(idx), static_cast<int>(height - 1)};
            else
                return {static_cast<int>(0), static_cast<int>(idx - width)};
        default:
            return {0, 0};
        }
    }

    ImgCoord next(Direction direction, ImgCoord &curr)
    {
        switch (direction)
        {
        case LeftToRight:
            return {curr.x + 1, curr.y};
        case TopLeftToBotRight:
            return {curr.x + 1, curr.y + 1};
        case TopToBot:
            return {curr.x, curr.y + 1};
        case TopRightToBotLeft:
            return {curr.x - 1, curr.y + 1};
        case RightToLeft:
            return {curr.x - 1, curr.y};
        case BotRightToTopLeft:
            return {curr.x - 1, curr.y - 1};
        case BotToTop:
            return {curr.x, curr.y - 1};
        case BotLeftToTopRight:
            return {curr.x + 1, curr.y - 1};
        default:
            return {0, 0};
        }
    }

    size_t maxSpan(Direction direction, size_t width, size_t height)
    {
        switch (direction)
        {
        case LeftToRight:
            return height;
        case TopLeftToBotRight:
            return height + width - 1;
        case TopToBot:
            return width;
        case TopRightToBotLeft:
            return height + width - 1;
        case RightToLeft:
            return height;
        case BotRightToTopLeft:
            return height + width - 1;
        case BotToTop:
            return width;
        case BotLeftToTopRight:
            return height + width - 1;
        default:
            return 0;
        }
    }

    bool inImage(ImgCoord &curr, size_t width, size_t height)
    {
        if (curr.x < 0 || curr.x >= width)
            return false;
        else if (curr.y < 0 || curr.y >= height)
            return false;
        else
            return true;
    }
};

float minLossAtPixel(float *pixelLoss)
{
    float minLoss = pixelLoss[0];
    for (size_t disparity = 1; disparity <= MAX_DISPARITY; ++disparity)
        minLoss = std::min(minLoss, pixelLoss[disparity]);
    return minLoss;
}

void calcLossesAtPixel(float *prevPixLoss, float *pixLoss, uint32_t *pixDist)
{
    float minLossAtPrevPixel = minLossAtPixel(prevPixLoss);

    for (size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity)
    {
        float smoothLoss = prevPixLoss[disparity];
        if (disparity > 0)
            smoothLoss = std::min(smoothLoss, prevPixLoss[disparity - 1] + P1);
        if (disparity < MAX_DISPARITY)
            smoothLoss = std::min(smoothLoss, prevPixLoss[disparity + 1] + P1);
        smoothLoss = std::min(smoothLoss, minLossAtPrevPixel + P2);
        pixLoss[disparity] = pixDist[disparity] + smoothLoss;
    }
}

void addDirLossToAggregateLoss(float *dirLoss, float *aggLoss)
{
    for (size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity)
        aggLoss[disparity] += dirLoss[disparity];
}

void directionalLossKernel(Direction::Direction direction,
                           Tensor<uint32_t> &distances,
                           Tensor<float> &aggLoss,
                           Tensor<float> &dirLoss,
                           size_t maxSpan)
{
    size_t width = distances.dims.cols;
    size_t height = distances.dims.rows;

    for (size_t idx = 0; idx < maxSpan; ++idx)
    {
        bool toggle = false;
        float *prevDirLoss = dirLoss.colPtr(idx, toggle);
        float *currDirLoss = dirLoss.colPtr(idx, !toggle);
        ImgCoord pixCoord = Direction::start(direction, idx, width, height);

        // Locations within the disparity arrays
        float *pixAggLoss = aggLoss.colPtr(pixCoord.y, pixCoord.x);

        // Initial row is equal to the hamming distances
        for (size_t disparity = 0; disparity <= MAX_DISPARITY + 1; ++disparity)
            prevDirLoss[disparity] = distances(pixCoord.y, pixCoord.x, disparity);

        addDirLossToAggregateLoss(prevDirLoss, pixAggLoss);
        pixCoord = Direction::next(direction, pixCoord);

        while (Direction::inImage(pixCoord, width, height))
        {
            pixAggLoss = aggLoss.colPtr(pixCoord.y, pixCoord.x);

            calcLossesAtPixel(prevDirLoss, currDirLoss, distances.colPtr(pixCoord.y, pixCoord.x));
            addDirLossToAggregateLoss(currDirLoss, pixAggLoss);

            toggle = !toggle;
            prevDirLoss = dirLoss.colPtr(idx, toggle);
            currDirLoss = dirLoss.colPtr(idx, !toggle);
            pixCoord = Direction::next(direction, pixCoord);
        }
    }
}

void disparityMapKernel(Tensor<Byte>& dispMap, Tensor<float>& loss,
                        size_t width, size_t height)
{
    for (size_t r = 0; r < height; ++r)
    {
        for (size_t c = 0; c < width; ++c)
        {
            ImgCoord pixCoord(c, r);

            size_t minDisparity = 0;
            float minLoss = loss(pixCoord.y, pixCoord.x, 0);
            for (size_t disparity = 1; disparity <= MAX_DISPARITY; ++disparity)
            {
                if (loss(pixCoord.y, pixCoord.x, disparity) < minLoss)
                {
                    minLoss = loss(pixCoord.y, pixCoord.x, disparity);
                    minDisparity = disparity;
                }
            }

            dispMap(pixCoord.y, pixCoord.x) = static_cast<Byte>(minDisparity);
        }
    }
}

Tensor<float> allocateAggregateLoss(size_t width, size_t height)
{
    TensorDims lossDims(height, width, MAX_DISPARITY + 1);
    Tensor<float> loss(lossDims);
    std::fill(loss.data, loss.data + (width*height*(MAX_DISPARITY+1)), 0.0);
    return loss;
}

Tensor<float> allocateDirectionalLoss(size_t maxSpan)
{
    TensorDims lossDims(maxSpan, 2, MAX_DISPARITY + 1);
    return {lossDims};
}

Tensor<Byte> allocateDisparityMap(size_t width, size_t height)
{
    TensorDims dispMapDims(height, width, 1);
    return {dispMapDims};
}

void lossInDirection(Direction::Direction direction, Tensor<uint32_t> &distances,
                     Tensor<float> &aggLoss)
{
    size_t maxSpan = Direction::maxSpan(direction, distances.dims.cols, distances.dims.rows);
    Tensor<float> dirLoss = allocateDirectionalLoss(maxSpan);

    directionalLossKernel(direction, distances, aggLoss, dirLoss, maxSpan);

    dirLoss.free();
}

Tensor<Byte> directionalLoss(Tensor<uint32_t> &distances)
{
    size_t width = distances.dims.cols;
    size_t height = distances.dims.rows;

    // distances: row x col x disparity
    // losses: row * col * disparity
    Tensor<float> aggLosses = allocateAggregateLoss(width, height);

    for (int direction = Direction::LeftToRight; direction < NUM_DIRECTIONS; ++direction)
        lossInDirection((Direction::Direction)direction, distances, aggLosses);

    Tensor<Byte> dispMap = allocateDisparityMap(width, height);

    disparityMapKernel(dispMap, aggLosses, width, height);
    aggLosses.free();

    return dispMap;
}