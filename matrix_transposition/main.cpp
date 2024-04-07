#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>

void transpose(int* src, int* dest, const int size, std::vector<long double>& times) {

    std::chrono::high_resolution_clock::time_point end;
    // start time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dest[j * size + i] = src[i * size + j];
        }
    }
    // end time
    end = std::chrono::high_resolution_clock::now();
    double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    executionTime /= 1e9;
    std::cout << "print time in transpose function: " << executionTime << " ns" << std::endl;
    times.push_back(executionTime);
}

void transponse_block(int* src, int* dest, const int size, std::vector<long double>& times, const int blockSize) {
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
    executionTime /= 1e9;
    std::cout << "print time in transpose function: " << executionTime << " ns" << std::endl;
    times.push_back(executionTime);
}

int* create_matrix(const int size) {
    int* matrix = new int[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = rand() % 100;
        }
    }
    return matrix;
}

void print_matrix(int* matrix, const int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

void calculate_effective_bandwidth(const std::vector<long double>& times, const int dimension) {
    const double total_data = (2 * dimension * dimension * sizeof(int)) / 1e9;
    std::cout << "Total data: " << total_data << " GB" << std::endl;
    std::vector<double> bandWidths;

    for (auto time : times) {
        std::cout << "Time: " << time << " ns" << std::endl;
        double bandwidth = total_data / time;
        bandwidth /= 1e9;
        bandWidths.push_back(bandwidth);
    }

    // calculate effective bandwidth in GB/s
    double meanBandwidth = std::accumulate(bandWidths.begin(), bandWidths.end(), 0.0) / bandWidths.size();
    double peakBandwidth = *std::max_element(bandWidths.begin(), bandWidths.end());
    std::cout << std::fixed << std::setprecision(11) << "Effective bandwidth (mean): " << meanBandwidth << " GB/s" << std::endl;
    std::cout << std::fixed << std::setprecision(11) << "Peak effective bandwidth: " << peakBandwidth << " GB/s" << std::endl;
}

int main(int argc, char const *argv[]) {
    // take integer from argv[1]
    // int dimension = pow(2, std::stoi(argv[1]));
    // std::array exponents = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::array exponents = {11};
    const int repeat = 100;
    const int blockSize = 16;
    // const std::string O_FLAG = "-O3";
    std::vector<long double> times;

    for (const int exp : exponents) {
        std::cout << "Dimension: " << exp << std::endl;
        int dimension = pow(2, exp);

        for (int i = 0; i < repeat; i++) {
            std::cout << "Iteration: " << i << std::endl;
            // create matrix
            int* matrix = create_matrix(dimension);
            int* transposed_matrix = new int[dimension * dimension];
            // transpose(matrix, transposed_matrix, dimension, times);
            transponse_block(matrix, transposed_matrix, dimension, times, blockSize);
            
            // print_matrix(matrix, dimension);
            // std::cout << "------------" << std::endl;
            // print_matrix(transposed_matrix, dimension);
            // std::cout << "------------" << std::endl;

            delete [] matrix;
            delete [] transposed_matrix;
        }

        // calculate effective bandwidth
        calculate_effective_bandwidth(times, dimension);
        

        // Write to file
        // std::ofstream file("output.csv", std::ios::app);
        // file << O_FLAG << "," << dimension << "," << meanBandwidth << "," << peakBandwidth << std::endl;
        // bandWidths.clear();
        times.clear();
    }
    


    return 0;
}
