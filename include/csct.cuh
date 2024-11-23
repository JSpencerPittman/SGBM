#ifndef CSCT_CUH_
#define CSCT_CUH_

#include<utility>

#include "image.h"

#define BLOCK_SIZE 16
#define RADIUS 1

struct CSCTResults {
    CSCTResults(bool* data, size_t numPixels, size_t compPerPix):
        data(data), numPixels(numPixels), compPerPix(compPerPix), 
        numBytes(numPixels * compPerPix * sizeof(bool)) {}

    bool* pixel(size_t pixelIdx) {
        return data + compPerPix * pixelIdx;
    }

    bool& comp(size_t pixelIdx, size_t compIdx) {
        return pixel(pixelIdx)[compIdx];
    }

    bool* data;
    size_t numPixels;
    size_t compPerPix;
    size_t numBytes;
};

// typedef std::pair<bool*, size_t> CSCTResults;

CSCTResults csct(Image& image);

#endif