#ifndef SGBM_SERIAL_H_
#define SGBM_SERIAL_H_

#include "config.h"
#include "serial/image.h"
#include "serial/tensor.h"

Tensor<bool> csct(Image &image);
Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT);

void sgbm(Image& leftImage, Image& rightImage);

#endif