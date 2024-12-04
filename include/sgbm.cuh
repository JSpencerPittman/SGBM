#ifndef SGBM_H_
#define SGBM_H_

#include "image.h"
#include "tensor.cuh"
#include "config.h"

Tensor<bool> csct(Image &image);
Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT);
Tensor<Byte> directionalLoss(Tensor<uint32_t>& distancesHost);

enum RankOperation {Minimum, Median, Maximum};
Tensor<Byte> rankFilter(Tensor<Byte> &dispMap, RankOperation operation);

Image sgbm(Image& leftImageHost, Image& rightImageHost);

#endif