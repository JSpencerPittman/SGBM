#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"

int main() {
    std::filesystem::path inPath = "../data/koala.jpg";
    Image koalaGray(inPath, true);
    koalaGray.writePng("koala_gray.png");
    Image koalaRGB(inPath);
    koalaRGB.writePng("koala_rgb.png");
    return 0;
}