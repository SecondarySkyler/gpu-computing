#include <iostream>
#include <random>
#include <chrono>


void print_matrix(int* matrix, const int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int* create_matrix(const int size) {
    int* matrix = new int[size * size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < size * size; i++) {
        matrix[i] = dis(gen);
    }

    return matrix;
}


template<typename T>
void matrix_multiplication(int exp, int rep) {
    const int size = pow(2, exp);
    const int dim = size * size;
    std::vector<double> executionTimes;
    const u_int32_t operations = (2 * pow(size, 3)) - pow(size, 2);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);
    // TODO: change code to generate matrix using create_matrix function
    for (int i = 0; i < rep; i++) {
        T* matrix_1 = new T[dim];
        T* matrix_2 = new T[dim];
        T* result = new T[dim];

        for (int i = 0; i < dim; i++) {
            matrix_1[i] = dis(gen);
            matrix_2[i] = dis(gen);
            result[i] = 0;
            
        }

        // Setup timers
        std::chrono::high_resolution_clock::time_point end;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int inner = 0; inner < size; inner++) {
                    result[i * size + j] += matrix_1[i * size + inner] * matrix_2[inner * size + j];
                }   
            }
        }

        end = std::chrono::high_resolution_clock::now();
        double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        executionTimes.push_back(executionTime);

        delete [] matrix_1;
        delete [] matrix_2;
        delete [] result;
    }

    // print_matrix(matrix_1, size);
    // std::cout << "------------" << std::endl;
    // print_matrix(matrix_2, size);
    // std::cout << "------------" << std::endl;
    // print_matrix(result, size);
    // std::cout << "------------" << std::endl;


    std::vector<double> executionFlops;
    // for each time we calculate the corresponding flops metric and we store it as MFlops
    for (auto time : executionTimes) {
        double const time_in_sec = time * pow(10, -9);
        double const flops = operations / time_in_sec;
        double const flops_in_mflops = flops * pow(10, -6);
        executionFlops.push_back(flops_in_mflops);
    }
    // Mean of flops
    double mean_mflops = std::accumulate(executionFlops.begin(), executionFlops.end(), 0.0) / rep;
    double peak_mflops = *std::max_element(executionFlops.begin(), executionFlops.end());


    std::cout << "Number of operations: " << operations << std::endl;
    std::cout << std::fixed << "Mean MFLOPS: " << mean_mflops << std::endl;
    std::cout << std::fixed << "Peak MFLOPS: " << peak_mflops << std::endl;

}



template<typename T, int B>
void tile_matrix_mul(int exp, int rep) {
    const int size = pow(2, exp);
    const int dim = size * size;
    std::vector<double> executionTimes;
    const int operations = (2 * pow(size, 3)) - pow(size, 2);

    for (int i = 0; i < rep; i++) {
        
        int* matrix_1 = create_matrix(size);
        int* matrix_2 = create_matrix(size);
        int* result = new int[dim] {0};

        // Setup timers
        std::chrono::high_resolution_clock::time_point end;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        // multiply matrix_1 and matrix_2 by blocks
        for (int i = 0; i < size; i += B) {
            for (int j = 0; j < size; j += B) {
                for (int k = 0; k < size; k += B) {

                    const int min_i = std::min(i + B, size);
                    const int min_j = std::min(j + B, size);
                    const int min_k = std::min(k + B, size);

                    for (int ii = i; ii < min_i; ii++) {
                        for (int jj = j; jj < min_j; jj++) {
                            for (int kk = k; kk < min_k; kk++) {
                                result[ii * size + jj] += matrix_1[ii * size + kk] * matrix_2[kk * size + jj];
                            }
                        }
                    }


                }
            }
        }

        end = std::chrono::high_resolution_clock::now();
        double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        executionTimes.push_back(executionTime);

        delete [] matrix_1;
        delete [] matrix_2;
        delete [] result;
    }

    // print_matrix(matrix_1, size);
    // std::cout << "------------" << std::endl;
    // print_matrix(matrix_2, size);
    // std::cout << "------------" << std::endl;
    // print_matrix(result, size);
    // std::cout << "------------" << std::endl;

    std::vector<double> executionFlops;
    // for each time we calculate the corresponding flops metric and we store it as MFlops
    for (auto time : executionTimes) {
        double const time_in_sec = time * pow(10, -9);
        double const flops = operations / time_in_sec;
        double const flops_in_mflops = flops * pow(10, -6);
        executionFlops.push_back(flops_in_mflops);
    }
    // Mean of flops
    double mean_mflops = std::accumulate(executionFlops.begin(), executionFlops.end(), 0.0) / rep;
    double peak_mflops = *std::max_element(executionFlops.begin(), executionFlops.end());


    std::cout << "Number of operations: " << operations << std::endl;
    std::cout << std::fixed << "Mean MFLOPS: " << mean_mflops << std::endl;
    std::cout << std::fixed << "Peak MFLOPS: " << peak_mflops << std::endl;
}



int main(int argc, char const *argv[]) {
    
    if (argc == 2) {
        int exponent = std::stoi(argv[1]);

        matrix_multiplication<int>(exponent, 10);
        // tile_matrix_mul<int, 16>(exponent, 10);
    } else {
        std::cout << "1 argument expected: exponent!" << std::endl;
    }
    
    return 0;
}
