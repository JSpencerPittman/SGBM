#include "cuda_image.cuh"

CudaImage::CudaImage(Image& image):
    m_path(image.path()), m_isGrayscale(image.isGrayscale()),
    m_width(image.width()), m_height(image.height()),
    m_channels(image.channels()) {
        cudaMalloc(&m_data, image.size());
        cudaMemcpy(m_data, image.data(), image.size(), cudaMemcpyHostToDevice);
}

CudaImage::~CudaImage() { cudaFree(m_data); }
