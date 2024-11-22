#ifndef IMAGE_H_
#define IMAGE_H_

#include <stddef.h>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

class Image
{
public:
    Image(const std::filesystem::path& path, bool grayscale = false);

private:
    static bool isValidFilePath(const std::filesystem::path &path, std::string &errorMessage) noexcept;

private:
    std::unique_ptr<stbi_uc[], void (*)(stbi_uc[])> m_data;

    std::string m_path;
    bool m_grayscale;
    size_t m_width;
    size_t m_height;
    size_t m_channels;
};

#endif
