#include "image.h"

#include <stdexcept>
#include <cstring>

#include <util/format.hpp>
#include "util/path_check.h"
#include "util/format.hpp"

namespace fs = std::filesystem;

Image::Image(const fs::path &path, bool grayscale)
    : m_path(path), m_isGrayscale(grayscale), m_data(nullptr)
{
    Status pathStatus = doesRegularFileExist(path);
    if (!pathStatus.succeed)
        throw std::invalid_argument(pathStatus.errorMessage.c_str());

    int width, height, origNumChannels;
    int desiredChannels = static_cast<int>(grayscale);

    Byte *stbData = stbi_load(path.c_str(), &width, &height, &origNumChannels, desiredChannels);
    size_t actualChannels = grayscale ? 1 : static_cast<size_t>(origNumChannels);
    m_data = reconstructSTBImageAsTensor(stbData, width, height, actualChannels);
    stbi_image_free(stbData);
}

Image::~Image() {
    m_data->free();
}

void Image::writePng(const fs::path &path) const
{
    // Verify the filename is valid
    Status filenameStatus = isValidFilePath(path, ".png");
    if (!filenameStatus.succeed)
        throw std::invalid_argument(filenameStatus.errorMessage.c_str());

    // Verify the parent path exists
    Status parentDirStatus = doesDirectoryExist(path.parent_path());
    if (!parentDirStatus.succeed)
        throw std::invalid_argument(parentDirStatus.errorMessage.c_str());

    stbi_write_png(path.c_str(), width(), height(), channels(), m_data->data, width() * channels());
}

std::unique_ptr<Tensor<Byte>> Image::reconstructSTBImageAsTensor(Byte stbData[],
                                                                 size_t width,
                                                                 size_t height,
                                                                 size_t channels)
{
    TensorDims imageDims(height, width, channels);
    size_t numValues = height * width * channels;
    Byte *tensorData = new Byte[numValues];
    std::memcpy(tensorData, stbData, numValues * sizeof(Byte));
    return std::make_unique<Tensor<Byte>>(imageDims, tensorData, false);
}
