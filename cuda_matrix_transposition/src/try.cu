#include <iostream>

#define dtype int

__global__ void naive_transpose(dtype* matrix, dtype* transposed_matrix, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width)
        transposed_matrix[row * width + col] = matrix[col * width + row];
}

dtype* create_matrix(int size) {
    int total_size = size * size;
    dtype* matrix = (dtype*)malloc(total_size * sizeof(dtype));
    for (int i = 0; i < total_size; i++) {
        matrix[i] = (dtype)(rand() % 100);
    }
    return matrix;
}

bool check_transpose(const dtype *gt, const dtype *result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (gt[i * size + j] != result[i * size + j]) {
                return false;
            }
        }
    }
    return true;
}


int main(int argc, char const *argv[])
{
    int N = 1 << 10;
    printf("Matrix size: %d\n", N);

    dim3 block(16, 16, 1);
    dim3 grid(N / block.x, N / block.y, 1);

    dtype* matrix = create_matrix(N);
    dtype* transposed_matrix = (dtype*)malloc(N * N * sizeof(dtype));

    dtype* d_matrix;
    dtype* d_transposed_matrix;
    dtype *ground_truth = (dtype*) malloc(N * N * sizeof(dtype));
            
    // calculate the ground truth
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ground_truth[j * N + i] = matrix[i * N + j];
        }
    }

    // // print the matrix
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", matrix[i * N + j]);
    //     }
    //     printf("\n");
    // }

    cudaMalloc(&d_matrix, N * N * sizeof(dtype));
    cudaMalloc(&d_transposed_matrix, N * N * sizeof(dtype));

    cudaMemcpy(d_matrix, matrix, N * N * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemset(d_transposed_matrix, 0, N * N * sizeof(dtype));

    naive_transpose<<<grid, block>>>(d_matrix, d_transposed_matrix, N);

    cudaDeviceSynchronize();

    cudaMemcpy(transposed_matrix, d_transposed_matrix, N * N * sizeof(dtype), cudaMemcpyDeviceToHost);

    // print the transposed matrix
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", transposed_matrix[i * N + j]);
    //     }
    //     printf("\n");
    // }

    if (check_transpose(ground_truth, transposed_matrix, N)) {
        printf("Matrix transposition is correct\n");
    } else {
        printf("Matrix transposition is incorrect\n");
    }

    cudaFree(d_matrix);
    cudaFree(d_transposed_matrix);
    free(matrix);
    free(transposed_matrix);
    return 0;
}
