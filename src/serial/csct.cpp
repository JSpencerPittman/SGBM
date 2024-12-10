#include "serial/sgbm.h"

#include <cmath>
#include <utility>

TensorCoord clampImageCoord(Image &image, int row, int col)
{
    return {
        static_cast<size_t>(std::max(std::min(row, static_cast<int>(image.height()) - 1), 0)),
        static_cast<size_t>(std::max(std::min(col, static_cast<int>(image.width()) - 1), 0)),
    };
}

std::pair<TensorCoord, TensorCoord> determineOpposers(Image &image, TensorCoord coordImage, int diffY, int diffX)
{
    int row = static_cast<int>(coordImage.row);
    int col = static_cast<int>(coordImage.col);
    return {
        clampImageCoord(image, row + diffY, col + diffX),
        clampImageCoord(image, row - diffY, col - diffX)};
}

void censusTransform(Image &image, Tensor<bool> &csctResults, TensorCoord coordImage)
{
    size_t compIdx = 0;
    int radiusInt = static_cast<int>(CSCT_RADIUS);
    for (int diffX = -radiusInt; diffX < 0; ++diffX)
    {
        for (int diffY = -radiusInt; diffY <= radiusInt; ++diffY)
        {
            auto opposers = determineOpposers(image, coordImage, diffY, diffX);
            csctResults(coordImage.row, coordImage.col, compIdx) =
                image.data()->value(opposers.first) >=
                image.data()->value(opposers.second);
            ++compIdx;
        }
    }
}

void csctKernel(Image &image, Tensor<bool> &csctResults)
{
    for (size_t r = 0; r < image.height(); ++r)
        for (size_t c = 0; c < image.width(); ++c)
            censusTransform(image, csctResults, {r, c});
};

Tensor<bool> allocateCSCTResultArray(Image &image)
{
    size_t diameter = 2 * CSCT_RADIUS + 1;
    size_t compPerPix = diameter * CSCT_RADIUS + CSCT_RADIUS;
    TensorDims csctResShape(image.height(), image.width(), compPerPix);
    return {csctResShape};
}

// PENDING
Tensor<bool> csct(Image &image)
{
    Tensor<bool> results = allocateCSCTResultArray(image);

    csctKernel(image, results);

    return results;
}
