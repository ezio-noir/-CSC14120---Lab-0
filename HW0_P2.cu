#include "utils.h"
#include <string.h>
#include <random>

enum version {
    HOST = 0,
    DEV_V1 = 1,
    DEV_V2 = 2,
};

/**
 * @brief Parses command-line arguments
 */
bool parseArguments(int argc, char **argv, int *ptr_n, int *ptr_v, char **ptr_filePath) {
    // Required parameters flags
    bool isNSet, isVSet;
    isNSet = isVSet = false;
    
    for (int i = 1; i < argc; i = i + 2) {
        if (strcmp(argv[i], "-n") == 0 && !isNSet) {
            *ptr_n = atoi(argv[i + 1]);
            isNSet = true;
        } else if (strcmp(argv[i], "-v") == 0 && !isVSet) {
            *ptr_v = atoi(argv[i + 1]);
            if (!(*ptr_v == HOST || *ptr_v == DEV_V1 || *ptr_v == DEV_V2)) {
                PANIC("-v expects value of 0, 1, or 2, but given %d.\n", *ptr_v);
            }
            isVSet = true;
        } else if (strcmp(argv[i], "-f") == 0) {
            size_t byteSize = (strlen(argv[i + 1]) + 1) * sizeof(char);
            *ptr_filePath = (char*)malloc(byteSize);
            memcpy(*ptr_filePath, argv[i + 1], byteSize);
        }
    }

    // Default version is to use host only
    if (!isVSet) {
        *ptr_v = HOST;
        isVSet = true;
    }

    return isNSet && isVSet;
}

/**
 * @brief Reads vectors' elements from file.
 */
void loadVectorsFromFile(float *vec1, float *vec2, int n, char *filePath) {
    FILE *f = fopen(filePath, "r");
    if (!f) {
        printf("Cannot read from file %s !\n", filePath);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; ++i) fscanf(f, "%f", vec1 + i);
    for (int i = 0; i < n; ++i) fscanf(f, "%f", vec2 + i);
    fclose(f);
}

/**
 * @brief Randomly initialize vectors elements.
 */
void initVectorsRandomly(float *vec1, float *vec2, int n) {
    srand(time(NULL));
    float denom = static_cast<float>(RAND_MAX);
    for (int i = 0; i < n; ++i) {
        vec1[i] = static_cast<float>(rand()) / denom;
        vec2[i] = static_cast<float>(rand()) / denom;
    }
}

void addVectorsHost(float *h_in1, float *h_in2, float *h_out, int n) {
    for (int i = 0; i < n; ++i) {
        h_out[i] = h_in1[i] + h_in2[i];
    }
}

__global__ void addVectorsDevV1(float *d_in1, float *d_in2, float *d_out, int n) {
    int idx1 = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (idx1 < n) d_out[idx1] = d_in1[idx1] + d_in2[idx1];

    int idx2 = idx1 + blockDim.x;
    if (idx2 < n) d_out[idx2] = d_in1[idx2] + d_in2[idx2];
}

__global__ void addVectorsDevV2(float *d_in1, float *d_in2, float *d_out, int n) {
    int idx1 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx1 < n) d_out[idx1] = d_in1[idx1] + d_in2[idx1];

    int idx2 = idx1 + 1;
    if (idx2 < n) d_out[idx2] = d_in1[idx2] + d_in2[idx2];
}

/**
 * @briefs Run add vectors on host and measure execution time
 * @return execution time in ms
 */
float h_performAddVectorsAndMeasureTime(float *h_in1, float *h_in2, float *h_out, int n, int version) {
    clock_t start, end;

    // Recording start and stop outside of case is to avoid duplication when we have many cases.
    // Since the switch statement is simple, the overhead should not affect the actual execution time.
    start = clock();
    switch (version) {
    case HOST:
        addVectorsHost(h_in1, h_in2, h_out, n);
        break;
    }
    end = clock();

    return (float)(end - start) / CLOCKS_PER_SEC * 1000;
}

float d_performAddVectorsAndMeasureTime(float *d_in1, float *d_in2, float *d_out, int n, int version, cudaEvent_t start, cudaEvent_t end) {
    dim3 blockSize(256);
    dim3 gridSize((n - 1) / blockSize.x + 1);

    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    switch (version) {
    case DEV_V1:
        addVectorsDevV1<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);
        break;
    case DEV_V2:
        addVectorsDevV2<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, n);
        break;
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsedTimeMs;
    cudaEventElapsedTime(&elapsedTimeMs, start, end);
    return elapsedTimeMs;

}

int main(int argc, char **argv) {
    if (!(argc == 3 || argc == 5 || argc == 7)) {
        printf("Expect 2 or 3 arguments.\n");
        exit(EXIT_FAILURE);
    }

    int n, v;
    char *filePath = NULL;

    if (!parseArguments(argc, argv, &n, &v, &filePath)) {
        printf("Invalid arguments!\n");
        exit(EXIT_FAILURE);
    }

    float *h_in1, *h_in2, *h_out;
    size_t vectorSizeInBytes = n * sizeof(float);

    // Allocate host memory
    MALLOC_CHECK(h_in1, float, vectorSizeInBytes);
    MALLOC_CHECK(h_in2, float, vectorSizeInBytes);
    MALLOC_CHECK(h_out, float, vectorSizeInBytes);

    // Initialize input vectors
    if (filePath) {
        printf("input size: %d ; version: %d ; input vectors: file %s ; ", n, v, filePath);
        loadVectorsFromFile(h_in1, h_in2, n, filePath);
    } else {
        printf("input size: %d ; version: %d ; input vectors: random ; ", n, v);
        initVectorsRandomly(h_in1, h_in2, n);
    }

    float elapsedTimeMs = 0;
    if (v == HOST) {
        elapsedTimeMs = h_performAddVectorsAndMeasureTime(h_in1, h_in2, h_out, n, v);
    } else if (v == DEV_V1 || v == DEV_V2) {
        float *d_in1, *d_in2, *d_out;
        CUDA_CHECK(cudaMalloc((void**) &d_in1, vectorSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**) &d_in2, vectorSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**) &d_out, vectorSizeInBytes));

        CUDA_CHECK(cudaMemcpy(d_in1, h_in1, vectorSizeInBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_in2, h_in2, vectorSizeInBytes, cudaMemcpyHostToDevice));

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));

        elapsedTimeMs = d_performAddVectorsAndMeasureTime(d_in1, d_in2, d_out, n, v, start, end);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));

        CUDA_CHECK(cudaFree(d_in1));
        CUDA_CHECK(cudaFree(d_in2));
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("elapsed time: %f ms.\n", elapsedTimeMs);

    // Free host memory
    free(h_in1);
    free(h_in2);
    free(h_out);

    return 0;
}