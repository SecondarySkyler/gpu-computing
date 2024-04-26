#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <map>
#include <optional>

#define O_FLAG "-O0"
#define dtype float
// this is used to set the times vector as optional
static std::vector<double> DEFAULT;

/**
 * This function is used to transpose the given matrix
 * with the naive algorithm
 * @param src: the source matrix
 * @param dest: the destination matrix (transposed)
 * @param dim: the dimension of the matrix, intended to be the length of the side
 * @param times: the vector to store the execution times
*/
void transpose(dtype* src, dtype* dest, const int dim, std::vector<double>& times = DEFAULT) {

    std::chrono::high_resolution_clock::time_point end;
    // start time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            dest[j * dim + i] = src[i * dim + j];
        }
    }
    // end time
    end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    executionTime /= 1e9; // convert to seconds
    times.push_back(executionTime);
}

/**
 * This function is used to transpose the given matrix
 * with the block algorithm
 * @param src: the source matrix
 * @param dest: the destination matrix (transposed)
 * @param size: the dimension of the matrix, intended to be the length of the side
 * @param blockSize: the size of the block
 * @param times: the vector to store the execution times
*/
void transponse_block(dtype* src, dtype* dest, const int size, const int blockSize, std::vector<double>& times = DEFAULT) {
    std::chrono::high_resolution_clock::time_point end;
    // start time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i += blockSize) {
        for (int j = 0; j < size; j += blockSize) {
            for (int k = i; k < i + blockSize; k++) {
                for (int l = j; l < j + blockSize; l++) {
                    dest[l * size + k] = src[k * size + l];
                }
            }
        }
    }
    // end time
    end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    executionTime /= 1e9; // convert to seconds
    times.push_back(executionTime);
}

/**
 * This function is used to create a matrix of size x size
 * 
 * @param size: the size of the matrix
 * @return matrix: the matrix created, filled with random numbers of type dtype
*/
dtype* create_matrix(const int size) {
    dtype* matrix = new dtype[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float rnd = static_cast <float> (rand()) / static_cast <float> (RAND_MAX / 100.0);
            matrix[i * size + j] = rnd;
        }
    }
    return matrix;
}

/**
 * This function prints the given matrix
 * 
 * @param matrix: the matrix to be printed
 * @param size: the size of the matrix
*/
void print_matrix(dtype* matrix, const int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}


/**
 * This main function is used to test the transpose_block function
 * 
*/
// int main(int argc, char const *argv[]) {
//     int dimension = 1024;
//     const int repeat = 100;
//     const int blockSize = 8;
//     // std::ofstream file("csv/cluster_block.csv", std::ios::app);
//     // std::array blockSizes = {2, 4, 8, 16, 32, 64};
//     std::vector<long double> times;

//     // for (const int blockSize : blockSizes) {
//         for (int i = 0; i < repeat; i++) {
//             int* matrix = create_matrix(dimension);
//             int* transposed_matrix = new int[dimension * dimension];
//             transponse_block(matrix, transposed_matrix, dimension, times, blockSize);

//             delete [] matrix;
//             delete [] transposed_matrix;
//         }

//         // for (auto time : times) {
//         //     file << O_FLAG << "," << dimension << "," << time << std::endl;   
//         // }

//         times.clear();
//     // }     
        
//     return 0;
// }

/**
 * This main function is used to test the naive transpose function
 * 
*/
// int main(int argc, char const *argv[]) {
//     // take integer from argv[1]
//     int dimension = pow(2, std::stoi(argv[1]));
//     std::ofstream file("csv/cluster_naive.csv", std::ios::app);
//     // std::array exponents = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     // std::array exponents = {2};
//     const int repeat = 100;
//     std::vector<double> times;

//     // for (const int exp : exponents) {
//     //     int dimension = pow(2, exp);
//         // std::cout << "Dimension: " << dimension << std::endl;

//         for (int i = 0; i < repeat; i++) {
//             // std::cout << "Iteration: " << i << std::endl;
//             // create matrix
//             int* matrix = create_matrix(dimension);
//             int* transposed_matrix = new int[dimension * dimension];
//             transpose(matrix, transposed_matrix, dimension, times);
            
//             // print_matrix(matrix, dimension);
//             // std::cout << "------------" << std::endl;
//             // print_matrix(transposed_matrix, dimension);
//             // std::cout << "------------" << std::endl;

//             delete [] matrix;
//             delete [] transposed_matrix;
//         }

//         // Write to file
//         // for (auto time : times) {
//         //     file << O_FLAG << "," << dimension << "," << time << std::endl;   
//         // }

//         times.clear();
//     // }
//     return 0;
// }

int main(int argc, char const *argv[]) {
    bool isNaive = true;
    srand(time(0));
    if (argc >= 2 || argc <= 3) {
        if (argv[2] != NULL) {
            std::string isBlock = argv[2];
            if (isBlock.find("yes") != std::string::npos) {
                isNaive = false;
            }
        }
        // take integer from argv[1]
        int side = std::stoi(argv[1]);
        int dimension = 1 << side;

        if (isNaive) {
            std::cout << "Naive Transpose" << std::endl;
            dtype* matrix = create_matrix(dimension);
            dtype* transposed_matrix = new dtype[dimension * dimension];
            transpose(matrix, transposed_matrix, dimension);
            print_matrix(matrix, dimension);
            std::cout << "------------" << std::endl;
            print_matrix(transposed_matrix, dimension);
            std::cout << "------------" << std::endl;
            
            delete [] matrix;
            delete [] transposed_matrix;
        } else {
            std::cout << "Block Transpose" << std::endl;
            const int blockSize = 4;
            dtype* matrix = create_matrix(dimension);
            dtype* transposed_matrix = new dtype[dimension * dimension];
            transponse_block(matrix, transposed_matrix, dimension, blockSize);
            print_matrix(matrix, dimension);
            std::cout << "------------" << std::endl;
            print_matrix(transposed_matrix, dimension);
            std::cout << "------------" << std::endl;

            delete [] matrix;
            delete [] transposed_matrix;
        }
    } else {
        std::cout << "Invalid number of arguments" << std::endl;
    }
    return 0;
}

