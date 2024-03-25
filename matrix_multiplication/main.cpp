#include <iostream>
#include <random>
#include <chrono>

template<typename T>
void matrix_multiplication(int exp, int rep) {
    const int size = pow(2, exp);
    std::vector<double> executionTimes;
    const int operations = (2 * pow(size, 3)) - pow(size, 2);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10);

    //for (int i = 0; i < rep; i++) {
        T matrix_1 [size][size];
        T matrix_2 [size][size];
        T result [size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix_1[i][j] = dis(gen);
                matrix_2[i][j] = dis(gen);
                result[i][j] = 0;
            }
        }

        // Setup timers
        std::chrono::high_resolution_clock::time_point end;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int inner = 0; inner < size; inner++) {
                    result[i][j] += matrix_1[i][inner] * matrix_2[inner][j];
                }   
            }
        }

        end = std::chrono::high_resolution_clock::now();
        double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        executionTimes.push_back(executionTime);


        
    //}



    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix_1[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "------------" << std::endl;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix_2[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "------------" << std::endl;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "------------" << std::endl;

    //double mean = std::accumulate(executionTimes.begin(), executionTimes.end(), 0.0) / rep;
    //std::cout << mean << std::endl;
    double time_in_sec = executionTimes[0] * pow(10, -9);
    double flops = operations / time_in_sec;
    double MFlops = flops * pow(10, -6);
    std::cout << "Number of operations: " << operations << std::endl;
    std::cout << "Execution Time: " << time_in_sec << "s" << std::endl;
    std::cout << std::fixed << "FLOPS: " << flops << std::endl;
    std::cout << std::fixed << "MFLOPS: " << MFlops << std::endl;

}

int main(int argc, char const *argv[]) {
    
    if (argc == 2) {
        int exponent = std::stoi(argv[1]);
        matrix_multiplication<int>(exponent, 20);
    } else {
        std::cout << "1 argument expected: exponent!" << std::endl;
    }
    
    return 0;
}
