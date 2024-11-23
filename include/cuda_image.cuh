#ifndef CUDA_IMAGE_H_
#define CUDA_IMAGE_H_

#include <stddef.h>
#include <string>
#include <filesystem>

#include "image.h"

namespace fs = std::filesystem;

class CudaImage
{
public:
    CudaImage(Image& image);
    ~CudaImage();

    /* -- Getters -- */
    std::string path() const { return m_path; }
    bool isGrayscale() const { return m_isGrayscale; }
    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    size_t channels() const { return m_channels; }

private:
    Byte* m_data;

    std::string m_path;
    bool m_isGrayscale;
    size_t m_width;
    size_t m_height;
    size_t m_channels;
};

#endif
