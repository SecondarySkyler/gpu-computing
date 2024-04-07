#include <stdio.h>
#include <stdlib.h>

__global__ void hello(void) {
    printf("Hello world from GPU, thread [%d, %d]\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char const *argv[]) {
    printf("Hello world from CPU\n");
    hello<<<1, 1024>>>();
    cudaDeviceSynchronize();
    return 0;
}
