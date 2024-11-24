#include <iostream>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "csct.cuh"
#include "hamming.cuh"

int main() {
    std::filesystem::path inPath = "../data/koala.jpg";
    Image image(inPath, true);

    CSCTResults leftRes = csct(image);
    CSCTResults rightRes = csct(image);

    printf("Part 1 complete\n");

    HamDistances hams = hamming(leftRes, rightRes, image.width(), image.height());
    
    for(size_t idx = 0; idx < 1000; ++idx) {
        std::cout << hams.data[idx] << " ";
    }

    delete [] leftRes.data;
    delete [] rightRes.data;
    delete [] hams.data;

    printf("It has been done.\n");

    return 0;
}