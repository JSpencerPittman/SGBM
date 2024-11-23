#ifndef IMAGE_H_
#define IMAGE_H_

#include <stddef.h>
#include <string>
#include <filesystem>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

namespace fs = std::filesystem;

typedef unsigned char Byte;
typedef std::unique_ptr<Byte[], void (*)(Byte[])> ImageUnqPtr;

class Image
{
public:
    Image(const fs::path &path, bool grayscale = false);

    void writePng(const fs::path &path) const;

    size_t size() const;

    /* -- Getters -- */
    Byte* data() { return m_data.get(); }

    std::string path() const { return m_path; }
    bool isGrayscale() const { return m_isGrayscale; }
    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    size_t channels() const { return m_channels; }

private:
    ImageUnqPtr m_data;

    std::string m_path;
    bool m_isGrayscale;
    size_t m_width;
    size_t m_height;
    size_t m_channels;
};

#endif
