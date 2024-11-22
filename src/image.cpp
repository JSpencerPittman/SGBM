#include "image.h"

#include <stdexcept>
#include <filesystem>
#include <string>
#include <util/format.hpp>

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <stb/stb_image_write.h>
//    stbi_write_png("./koala_gray.png", x, y, 1, data, x);

Image::Image(const std::filesystem::path &path, bool grayscale)
    : m_path(path), m_grayscale(grayscale),
      m_data(nullptr, reinterpret_cast<void (*)(stbi_uc[])>(stbi_image_free))
{
    std::string errorMessage;
    if (!isValidFilePath(path, errorMessage))
        throw std::invalid_argument(errorMessage.c_str());

    int x, y, n;
    int desired_channels = static_cast<int>(grayscale);

    m_data = std::unique_ptr<stbi_uc[], void(*)(stbi_uc[])>(
        stbi_load(path.c_str(), &x, &y, &n, desired_channels),
        reinterpret_cast<void(*)(stbi_uc[])>(stbi_image_free)
    );

    m_width = static_cast<size_t>(x);
    m_height = static_cast<size_t>(y);
    m_channels = grayscale ? 1 : static_cast<size_t>(n);
}

bool Image::isValidFilePath(const std::filesystem::path &imagePath, std::string &errorMessage)
{
    if (!std::filesystem::exists(imagePath))
    {
        std::string format = "Provided path does not exists: %s";
        errorMessage = stringFormat(format, imagePath.c_str());
        return false;
    }
    else if (!std::filesystem::is_regular_file(imagePath))
    {
        std::string format = "Provided path is not a regular file: %s";
        errorMessage = stringFormat(format, imagePath.c_str());
        return false;
    }
    return true;
}