#include <iostream>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "csct.cuh"

int main() {
    std::filesystem::path inPath = "../data/koala.jpg";
    Image image(inPath, true);

    CSCTResults res = csct(image);

    std::cout << "IMAGESIZE:" << image.width() << " x " << image.height() << std::endl;

    for(size_t idx = 0; idx < image.width(); ++idx) {
        for(size_t idx2 = 0; idx2 < 4; ++idx2)
            std::cout << static_cast<int>(res.data[idx*4 + idx2]);
        std::cout << " ";
    }
    std::cout << std::endl;

    delete [] res.data;

    printf("It has been done.\n");

    return 0;
}