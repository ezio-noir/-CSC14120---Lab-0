#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

#define PANIC(format, ...) {                                                            \
    printf(format, __VA_ARGS__);                                                        \
    exit(EXIT_FAILURE);                                                                 \
}                                                                                       \

#define CUDA_CHECK(call) {                                                              \
    cudaError_t err = call;                                                             \
    if (err != cudaSuccess) {                                                           \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}                                                                                       \

#define MALLOC_CHECK(ptr, type, size) {                                                 \
    ptr = (type*)malloc(size);                                                          \
    if (ptr == NULL) {                                                                  \
        printf("Memory allocation failed.\n");                                          \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}                                                                                       \

#endif