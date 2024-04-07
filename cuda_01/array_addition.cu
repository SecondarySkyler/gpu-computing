#include <iostream>

__global__ void array_add(int* array_1, int* array_2, int* out, int N) {
    int i = threadIdx.x;
    int stride = blockDim.x;
    int iter = N / stride;
    // avoid out of bounds if I have more threads than elements
    if (i >= N) {
        return;
    }

    // for (int i = index; i < N; i += stride) {
    //     out[i] = array_2[i] + array_1[i];
    //     // i += stride;
    //     // avoid out of bounds if i have more threads than elements
    //     // if (i >= N) {
    //     //     return;
    //     // }
    // }

    for (int j = 0; j < iter; j++) {
        out[i] = array_2[i] + array_1[i];
        i += stride;

        // avoid out of bounds if i have more threads than elements
        if (i >= N) {
            return;
        }
    }
    
    
}

int main(int argc, char const *argv[]) {
    int N = 1 << 11;
    int *array_1, *array_2, *output_array;

    cudaMallocManaged(&array_1, N * sizeof(int));
    cudaMallocManaged(&array_2, N * sizeof(int));
    cudaMallocManaged(&output_array, N * sizeof(int));
    // setup cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize arrays with random numbers
    for (int i = 0; i < N; i++) {
        array_1[i] = rand() % 10;
        array_2[i] = rand() % 10;
    }
    

    // record event start
    cudaEventRecord(start);

    // launch kernel
    array_add<<<1, N>>>(array_1, array_2, output_array, N);
    cudaDeviceSynchronize();

    // record event stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    // calculate time
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel Time: " << milliseconds << " ms" << std::endl;


    // free memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(array_1);
    cudaFree(array_2);
    cudaFree(output_array);
    return 0;
}
