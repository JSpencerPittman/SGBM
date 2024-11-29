#include <stdio.h>
#include <cstdint>
#include <algorithm>

typedef uint8_t Byte;

void quicksort(Byte* arr, size_t len) {
    if(len <= 1) return;
    Byte pivot = arr[0];
    size_t pivotIdx = 0;
    for(size_t idx = 1; idx < len; ++idx) {
        if(arr[idx] <= pivot) {
            arr[pivotIdx] = arr[idx];
            arr[idx] = arr[pivotIdx+1];
            arr[++pivotIdx] = pivot;
        }
    }
    quicksort(arr, pivotIdx);
    quicksort(arr + pivotIdx + 1, len - pivotIdx - 1);
}

void printArray(Byte* arr, size_t len) {
    for(size_t idx = 0; idx < len; ++idx) {
        printf("%u, ", arr[idx]);
    }
    printf("\n");
}

int main() {
    Byte sample[] = {3, 4, 0, 1, 2};
    quicksort(sample, 5);
    printArray(sample, 5);
}