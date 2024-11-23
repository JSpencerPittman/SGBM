#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "sgbm.cuh"

int main() {
    std::filesystem::path inPath = "../data/koala.jpg";
    Image koalaGray(inPath, true);

    sgbm(koalaGray);

    return 0;
}