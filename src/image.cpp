#include "image.h"

#include <stdexcept>
#include <util/format.hpp>

#include "util/path_check.h"
#include "util/format.hpp"

#include<iostream>

namespace fs = std::filesystem;


Image::Image(const fs::path &path, bool grayscale)
    : m_path(path), m_grayscale(grayscale),
      m_data(nullptr, reinterpret_cast<void (*)(stbi_uc[])>(stbi_image_free))
{
    Status pathStatus = doesRegularFileExist(path);
    if (!pathStatus.succeed)
        throw std::invalid_argument(pathStatus.errorMessage.c_str());

    int width, height, origNumChannels;
    int desiredChannels = static_cast<int>(grayscale);

    m_data = std::unique_ptr<stbi_uc[], void(*)(stbi_uc[])>(
        stbi_load(path.c_str(), &width, &height, &origNumChannels, desiredChannels),
        reinterpret_cast<void(*)(stbi_uc[])>(stbi_image_free)
    );

    m_width = static_cast<size_t>(width);
    m_height = static_cast<size_t>(height);
    m_channels = grayscale ? 1 : static_cast<size_t>(origNumChannels);
}

void Image::writePng(const fs::path& path) const {
    // Verify the filename is valid
    Status filenameStatus = isValidFilePath(path, ".png");
    if(!filenameStatus.succeed) throw std::invalid_argument(filenameStatus.errorMessage.c_str());

    // Verify the parent path exists
    Status parentDirStatus = doesDirectoryExist(path.parent_path());
    if(!parentDirStatus.succeed) throw std::invalid_argument(parentDirStatus.errorMessage.c_str());

    stbi_write_png(path.c_str(), m_width, m_height, m_channels, m_data.get(), m_width * m_channels);
}
