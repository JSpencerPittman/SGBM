#ifndef IMAGE_H_
#define IMAGE_H_

#include <stddef.h>
#include <string>
#include <filesystem>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

namespace fs = std::filesystem;

class Image
{
public:
    Image(const fs::path& path, bool grayscale = false);

    void writePng(const fs::path& path) const;

private:
    std::unique_ptr<stbi_uc[], void (*)(stbi_uc[])> m_data;

    std::string m_path;
    bool m_grayscale;
    size_t m_width;
    size_t m_height;
    size_t m_channels;
};

#endif
