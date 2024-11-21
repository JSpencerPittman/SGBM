#include <iostream>

#include "kernel.cuh"

int main() {
    std::cout << "Host: Starting" << std::endl;
    hello_world();
    std::cout << "Host: Terminating" << std::endl;
    return 0;
}