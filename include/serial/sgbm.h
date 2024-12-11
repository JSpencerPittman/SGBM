#ifndef SGBM_SERIAL_H_
#define SGBM_SERIAL_H_

#include "config.h"
#include "serial/image.h"
#include "serial/tensor.h"

Tensor<bool> csct(Image &image);
Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT);
Tensor<Byte> directionalLoss(Tensor<uint32_t>& distancesHost);

enum RankOperation {Minimum, Median, Maximum};
Tensor<Byte> rankFilter(Tensor<Byte> &dispMap, RankOperation operation);

Image sgbm(Image& leftImage, Image& rightImage);

#endif