#ifndef CSCT_CUH_
#define CSCT_CUH_

#include<utility>

#include "image.h"

#define BLOCK_SIZE 16
#define RADIUS 1

typedef std::pair<bool*, size_t> CSCTResults;

CSCTResults csct(Image& image);

#endif