#ifndef DIRECTIONAL_H_
#define DIRECTIONAL_H_

#include<utility>
#include<cuda_runtime.h>

#include "hamming.cuh"

#define P1 7
#define P2 17

typedef Tensor<float> Loss;

FlatImage directionalLoss(Distances& distancesHost);

#endif