#ifndef SGBM_H_
#define SGBM_H_

#include "image.h"
#include "tensor.cuh"
#include "config.h"

Tensor<bool> csct(Image &image);
Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT);
Tensor<Byte> directionalLoss(Tensor<uint32_t>& distancesHost);
Tensor<Byte> medianBlur(Tensor<Byte> & dispMap);

Image sgbm(Image& leftImageHost, Image& rightImageHost);

#endif