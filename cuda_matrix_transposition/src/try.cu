#include <iostream>

#define dtype int

const int TILE_DIM = 16;
const int REPEAT = 100;

__global__ void naive_transpose(dtype* matrix, dtype* transposed_matrix, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width)
        transposed_matrix[row * width + col] = matrix[col * width + row];
}

__global__ void transpose_coalesced(dtype* matrix, dtype* transposed_matrix, int width) {
    __shared__ dtype tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;

    // load the matrix into shared memory
    if (row < width && col < width) {
        tile[threadIdx.y][threadIdx.x] = matrix[col * width + row];
    }

    __syncthreads();

    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;

    if (row < width && col < width) {
        transposed_matrix[col * width + row] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_coalesced_no_bank_conflicts(dtype* matrix, dtype* transposed_matrix, int width) {
    __shared__ dtype tile[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;

    // load the matrix into shared memory
    if (row < width && col < width) {
        tile[threadIdx.y][threadIdx.x] = matrix[col * width + row];
    }

    __syncthreads();

    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;

    if (row < width && col < width) {
        transposed_matrix[col * width + row] = tile[threadIdx.x][threadIdx.y];
    }
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

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

int main(int argc, char const *argv[])
{
    int N = 1 << 10;
    printf("Matrix size: %d\n", N);

    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid(N / block.x, N / block.y, 1);

    dim3 grid_v2(
        (N + TILE_DIM - 1) / TILE_DIM,
        (N + TILE_DIM - 1) / TILE_DIM,
        1
    );


    int mem_size = N * N * sizeof(dtype);

    // Define a new type to create an array of pointers to transposition functions
    typedef void (*transpose_func)(dtype*, dtype*, int);
    transpose_func transpose_functions[] = {naive_transpose, transpose_coalesced, transpose_coalesced_no_bank_conflicts};
    const char* function_names[] = {"Naive Transpose", "Coalesced Transpose", "Coalesced Transpose no Bank Conflicts"};
    uint16_t length = sizeof(transpose_functions) / sizeof(transpose_functions[0]);

    // Iterate over the functions
    for (uint16_t f = 0; f < length; f++) {

        dtype* matrix = create_matrix(N);
        dtype* transposed_matrix = (dtype*)malloc(N * N * sizeof(dtype));
        dtype *ground_truth = (dtype*) malloc(N * N * sizeof(dtype));
        
        // calculate the ground truth
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                ground_truth[j * N + i] = matrix[i * N + j];
            }
        }

        dtype* d_matrix;
        dtype* d_transposed_matrix;


        cudaMalloc(&d_matrix, N * N * sizeof(dtype));
        cudaMalloc(&d_transposed_matrix, N * N * sizeof(dtype));

        cudaMemcpy(d_matrix, matrix, N * N * sizeof(dtype), cudaMemcpyHostToDevice);
        cudaMemset(d_transposed_matrix, 0, N * N * sizeof(dtype));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        warm_up_gpu<<<1, 1>>>();

        cudaEventRecord(start, 0);
        for (int i = 0; i < REPEAT; i++) 
            transpose_functions[f]<<<grid, block>>>(d_matrix, d_transposed_matrix, N);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        cudaDeviceSynchronize();

        cudaMemcpy(transposed_matrix, d_transposed_matrix, N * N * sizeof(dtype), cudaMemcpyDeviceToHost);

        if (check_transpose(ground_truth, transposed_matrix, N)) {
            printf("Matrix transposition is correct\n");
            printf("Algorithm: %s\n", function_names[f]);
            printf("Bandwidth: %20.2f\n", 2 * mem_size * 1e-6 * REPEAT / ms );
            printf("--------------------------------\n");
        } else {
            printf("Matrix transposition is incorrect\n");
            printf("--------------------------------\n");
        }


        cudaFree(d_matrix);
        cudaFree(d_transposed_matrix);
        free(matrix);
        free(transposed_matrix);
    }

    return 0;
}
