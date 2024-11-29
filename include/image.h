#ifndef IMAGE_H_
#define IMAGE_H_

#include <stddef.h>
#include <string>
#include <filesystem>
#include <optional>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "tensor.cuh"

namespace fs = std::filesystem;

typedef unsigned char Byte;
typedef std::unique_ptr<Byte[], void (*)(Byte[])> ImageUniquePtr;

class Image
{
public:
    Image(const fs::path &path, bool grayscale = false);

    void writePng(const fs::path &path) const;

    /* -- Getters -- */
    std::optional<std::string> path() const { return m_path; }
    bool isGrayscale() const { return m_isGrayscale; }
    size_t width() const { return m_data->dims.cols; }
    size_t height() const { return m_data->dims.rows; }
    size_t channels() const { return m_data->dims.channels; }
    size_t bytes() const { return m_data->bytes(); };

private:
    // Copy source image
    static Tensor<Byte>* reconstructSTBImageAsTensor(Byte stbData[], size_t width, size_t height, size_t channel);

private:
    Tensor<Byte>* m_data;
    std::optional<std::string> m_path;
    bool m_isGrayscale;
};

#endif
