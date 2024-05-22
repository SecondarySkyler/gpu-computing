#include <iostream>

#define dtype int

__global__ naive_transpose(dtype* matrix, dtype* transposed_matrix, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < size && col < size) {
        transposed_matrix[row * size + col] = matrix[col * size + row];
    }
}

dtype* create_matrix(int size) {
    dtype* matrix = new dtype[size * size];
    for (int i = 0; i < size * size; i++) {
        matrix[i] = i;
    }
    return matrix;
}

int main(int argc, char const *argv[]) {
    // if the argc is less than 2, then print the usage and exit
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " matrix dimension" << std::endl;
        return 1;
    }

    // get the matrix dimension from the command line argument
    int dimension = atoi(argv[1]);
    int matrix_size = 1 << dimension;

    // create the matrices
    dtype* matrix = create_matrix(matrix_size);
    dtype* transposed_matrix = new dtype[matrix_size * matrix_size];

    // allocate the memory on the device
    dtype* src_matrix, dst_matrix;
    checkCuda( cudaMalloc(&src_matrix, matrix_size) )
    checkCuda( cudaMalloc(&dst_matrix, matrix_size) )

    // copy the matrix to the device
    checkCuda( cudaMemcpy(src_matrix, matrix, matrix_size, cudaMemcpyHostToDevice) )


    return 0;
}
