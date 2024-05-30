#include <iostream>
#include <fstream>

#define dtype int

const int TILE_DIM = 32;
const int BLOCK_ROWS = 16;
const int REPETITIONS = 100;

__global__ void naive_transpose(dtype* matrix, dtype* transposed_matrix) {
    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        transposed_matrix[row * width + (col + i)] = matrix[(col + i) * width + row];
    }
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
__global__ void transposeCoalesced(dtype* matrix, dtype* transposed_matrix)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     transposed_matrix[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transposeCoalescedNoBankConflicts(dtype* matrix, dtype* transposed_matrix)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     transposed_matrix[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void naive_transpose_v2(dtype* matrix, dtype* transposed_matrix, int TILE_DIM, int BLOCK_ROWS) {
    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        transposed_matrix[row * width + (col + i)] = matrix[(col + i) * width + row];
    }
}

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
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
// TODO: use rand to generate random matrices
dtype* create_matrix(int size) {
    int total_size = size * size;
    dtype* matrix = (dtype*)malloc(total_size * sizeof(dtype));
    for (int i = 0; i < total_size; i++) {
        matrix[i] = (dtype)(rand() % 100);
    }
    return matrix;
}

int main(int argc, char const *argv[]) {
    // if the argc is less than 2, then print the usage and exit
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " matrix dimension" << std::endl;
        return 1;
    }

    /*
    -------- DEVICE PROPERTIES --------
    */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    // get the matrix dimension from the command line argument
    int dimension = atoi(argv[1]);
    printf("Exponent: %d\n", dimension);
    int side_length = 1 << dimension;
    printf("Matrix size: %d X %d\n", side_length, side_length);
    const int matrix_size = side_length * side_length;
    const int mem_size = matrix_size * sizeof(dtype);

    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid(side_length / TILE_DIM, side_length / TILE_DIM, 1);

    typedef void (*transposing_functions) (dtype*, dtype*);
    transposing_functions fncts[] = {naive_transpose, transposeCoalesced, transposeCoalescedNoBankConflicts};
    const char *transposing_function_names[] = {"Naive transpose", "Coalesced transpose", "Coalesced transpose w/o bank conflicts"};
    uint16_t tf_length = sizeof(fncts) / sizeof(fncts[0]);
    // iterate over the transposing functions
    for (int f = 0; f < tf_length; f++) {

        // repeat the experiment 100 times
        // for (int i = 0; i < REPETITIONS; i++) {
            
            // create the matrices
            dtype *matrix = create_matrix(side_length);
            dtype *transposed_matrix = (dtype*) malloc(mem_size);
            dtype *ground_truth = (dtype*) malloc(mem_size);
            
            // calculate the ground truth
            for (int i = 0; i < side_length; i++) {
                for (int j = 0; j < side_length; j++) {
                    ground_truth[j * side_length + i] = matrix[i * side_length + j];
                }
            }
            
            // allocate the memory on the device
            dtype *src_matrix, *dst_matrix;
            cudaMalloc(&src_matrix, mem_size);
            cudaMalloc(&dst_matrix, mem_size);

            // copy the matrix to the device
            cudaMemcpy(src_matrix, matrix, mem_size, cudaMemcpyHostToDevice);
            cudaMemset(dst_matrix, 0, mem_size);
        
            // events
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            float ms;

            // warm up the GPU
            warm_up_gpu<<<1, 1>>>();

            cudaEventRecord(start, 0);
            for (int i = 0; i < REPETITIONS; i++)
                fncts[f]<<<grid, block>>>(src_matrix, dst_matrix);
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);

            cudaDeviceSynchronize(); // is this necessary?

            // copy the transposed matrix back to the host
            cudaMemcpy(transposed_matrix, dst_matrix, mem_size, cudaMemcpyDeviceToHost);

            // check if the transposed matrix is correct
            if (check_transpose(ground_truth, transposed_matrix, side_length)) {
                printf("Matrix transposition is correct\n");
                printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
                printf("%25s", transposing_function_names[f]);
                printf("%20.2f\n", 2 * mem_size * 1e-6 * REPETITIONS / ms );
            } else {
                printf("Error in matrix transposition\n");
            }

            // free the memory
            cudaFree(src_matrix);
            cudaFree(dst_matrix);
            free(matrix);
            free(transposed_matrix);
        // }
    }

    return 0;
}

// int main(int argc, char const *argv[]) {
    
//     // open file to write the results
//     std::ofstream file("csv/results.csv", std::ios::app);

    
//     for (int i = 1; i <= 12; i++) {
//         int side_length = 1 << i; // 2^i => 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
//         int matrix_size = side_length * side_length;
//         int mem_size = matrix_size * sizeof(dtype);

//         for (int j = 2; j <= side_length; j *= 2) {
//             if (j > 1024) // this is sus maybe we can set the for loop to arrive at 1024
//                 break;
            
//             for (int k = 1; k <= j; k *= 2) {
//                 int TILE_DIM = j;
//                 int BLOCK_ROWS = k;

//                 dim3 grid(side_length / TILE_DIM, side_length / TILE_DIM, 1);
//                 dim3 block(TILE_DIM, BLOCK_ROWS, 1);

//                 // create the matrices
//                 dtype *matrix = create_matrix(side_length);
//                 dtype *transposed_matrix = (dtype*) malloc(mem_size);
//                 dtype *ground_truth = (dtype*) malloc(mem_size);

//                 // calculate the ground truth
//                 for (int i = 0; i < side_length; i++) {
//                     for (int j = 0; j < side_length; j++) {
//                         ground_truth[j * side_length + i] = matrix[i * side_length + j];
//                     }
//                 }

//                 // allocate the memory on the device
//                 dtype *src_matrix, *dst_matrix;
//                 cudaMalloc(&src_matrix, mem_size);
//                 cudaMalloc(&dst_matrix, mem_size);

//                 // copy the matrix to the device
//                 cudaMemcpy(src_matrix, matrix, mem_size, cudaMemcpyHostToDevice);
//                 cudaMemset(dst_matrix, 0, mem_size);

//                 // events
//                 cudaEvent_t start, stop;
//                 cudaEventCreate(&start);
//                 cudaEventCreate(&stop);
//                 float ms;

//                 // warm up the GPU
//                 warm_up_gpu<<<1, 1>>>();

//                 cudaEventRecord(start, 0);
//                 for (int i = 0; i < REPETITIONS; i++)
//                     naive_transpose_v2<<<grid, block>>>(src_matrix, dst_matrix, TILE_DIM, BLOCK_ROWS);
                
//                 cudaEventRecord(stop, 0);
//                 cudaEventSynchronize(stop);
//                 cudaEventElapsedTime(&ms, start, stop);

//                 cudaDeviceSynchronize(); // is this necessary?

//                 // copy the transposed matrix back to the host
//                 cudaMemcpy(transposed_matrix, dst_matrix, mem_size, cudaMemcpyDeviceToHost);

//                 // check if the transposed matrix is correct
//                 if (check_transpose(ground_truth, transposed_matrix, side_length)) {
//                     // printf("Matrix transposition is correct\n");
//                     // printf("%25s%25s%25s%25s\n", "Dimension", "TILE_DIM", "BLOCK_ROWS", "Bandwidth (GB/s)");
//                     // printf("%25d%25d%25d%20.2f\n", side_length, TILE_DIM, BLOCK_ROWS, 2 * mem_size * 1e-6 * REPETITIONS / ms );
//                     file << side_length << "," << TILE_DIM << "," << BLOCK_ROWS << "," << 2 * mem_size * 1e-6 * REPETITIONS / ms << std::endl;
//                 } 
//                 // else {
//                 //     printf("Error in matrix transposition\n");
//                 //     printf("Dimension: %d, TILE_DIM: %d, BLOCK_ROWS: %d  \n", side_length, TILE_DIM, BLOCK_ROWS);
//                 // }

//                 // free the memory
//                 cudaEventDestroy(start);
//                 cudaEventDestroy(stop);
//                 cudaFree(src_matrix);
//                 cudaFree(dst_matrix);
//                 free(matrix);
//                 free(transposed_matrix);
//             }
//         }
//     }
//     return 0;
// }

