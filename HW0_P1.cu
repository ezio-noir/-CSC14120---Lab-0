#include <stdio.h>
#include "utils.h"

void printAllGPUsSpecs() {
    int devCount;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));

    printf("Found %d compute-capable GPUs.\n", devCount);
    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp devProp;
        CUDA_CHECK(cudaGetDeviceProperties(&devProp, i));
        printf("\t* GPU#%02d:\n", i);
        printf("\t\t- Name: %s\n", devProp.name);
        printf("\t\t- Computation capabilities: %d.%d\n", devProp.major, devProp.minor);
        printf("\t\t- Maximum number of block dimensions: (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("\t\t- Maximum number of grid dimensions: (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("\t\t- Maximum size of GPU memory: %zuB ~ %.2fGB\n", devProp.totalGlobalMem, static_cast<float>(devProp.totalGlobalMem) / static_cast<float>(1 << 30));
        printf("\t\t- Constant memory size: %zuB ~ %.2fKB\n", devProp.totalConstMem, static_cast<float>(devProp.totalConstMem) / static_cast<float>(1 << 10));
        printf("\t\t- Shared memory size: %zuB ~ %.2fKB\n", devProp.sharedMemPerBlock, static_cast<float>(devProp.sharedMemPerBlock) / static_cast<float>(1 << 10));
        printf("\t\t- Warp size: %d\n", devProp.warpSize);
    }
}

int main(int argc, char **argv) {
    printAllGPUsSpecs();

    return 0;
}