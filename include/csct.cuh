#ifndef CSCT_CUH_
#define CSCT_CUH_

#include <utility>
#include <cuda_runtime.h>

#include "image.h"
#include "tensor.cuh"

#define BLOCK_SIZE 32
#define RADIUS 2

typedef Tensor<Byte> FlatImage;
typedef Tensor<bool> CSCTResults;

CSCTResults csct(FlatImage &image);

#endif