#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>


void transpose(int* src, int* dest, const int size, std::vector<double>& times) {

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

int main(int argc, char const *argv[]) {
    // take integer from argv[1]
    int dimension = pow(2, std::stoi(argv[1]));
    const int repeat = 100;
    const std::string O_FLAG = "-O0";
    std::vector<double> times;

    for (int i = 0; i < repeat; i++) {
        std::cout << "Iteration: " << i << std::endl;
        // create matrix
        int* matrix = create_matrix(dimension);
        int* transposed_matrix = new int[dimension * dimension];
        transpose(matrix, transposed_matrix, dimension, times);

        // print_matrix(matrix, dimension);
        // std::cout << "------------" << std::endl;
        // print_matrix(transposed_matrix, dimension);
        // std::cout << "------------" << std::endl;

        delete [] matrix;
        delete [] transposed_matrix;
    }
    // calculate effective bandwidth in GB/s
    const int total_data = (2 * dimension * dimension * sizeof(int));
    std::vector<double> bandWidths(repeat);
    for (auto time : times) {
        double const time_in_sec = time * pow(10, -9);
        double const bandwidth = (total_data / time_in_sec) * pow(10, -9);
        bandWidths.push_back(bandwidth);
    }
    
    double meanBandwidth = std::accumulate(bandWidths.begin(), bandWidths.end(), 0.0) / bandWidths.size();
    double peakBandwidth = *std::max_element(bandWidths.begin(), bandWidths.end());
    std::cout << std::fixed;
    std::cout << std::setprecision(2) << "Effective bandwidth (mean): " << meanBandwidth << " GB/s" << std::endl;
    std::cout << "Peak effective bandwidth: " << peakBandwidth << " GB/s" << std::endl;

    // Write to file
    std::ofstream file("output.csv", std::ios::app);
    file << O_FLAG << "," << dimension << "," << meanBandwidth << "," << peakBandwidth << std::endl;

    return 0;
}
