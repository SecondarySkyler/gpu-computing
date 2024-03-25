#include <iostream>
#include <fstream>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <random>
#include <type_traits>


// Templating a random generator
template<typename T, typename R = std::mt19937_64>
T random(const T& min, const T& max) {
    R rng{std::random_device{}()};
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng);
    } else if (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rng);
    } else {
        return T{};
    }
}

double calculate_mean(const std::vector<double>& times) {
    int reps = times.size();
    return std::accumulate(times.begin(), times.end(), 0.0) / reps;
}

double calculate_variance(const std::vector<double>& times, const int mean) {
    double variance = 0.0;
    for (auto time : times) {
        double aux = time - mean;
        variance += pow(aux, 2);
    }
    return variance / times.size();
}

template <class T>
void array_addition(int exp, int rep) {
    int arraySize = pow(2, exp);
    std::vector<double> executionTimes;
    
    for (int i = 0; i < rep; i++) {
        T* array_1 = new T[arraySize];
        T* array_2 = new T[arraySize];
        T* result = new T[arraySize];

        for (int i = 0; i < arraySize; i++) {
            array_1[i] = random<T>(0, 100);
            array_2[i] = random<T>(0, 100);
        }

        // Setup timers
        std::chrono::high_resolution_clock::time_point end;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < arraySize; i++) {
            result[i] = array_1[i] + array_2[2];
        }

        end = std::chrono::high_resolution_clock::now();
        
        double executionTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        executionTimes.push_back(executionTime);

        delete [] array_1;
        delete [] array_2;
        delete [] result;

    }

    // Statistics calculation
    double mean = calculate_mean(executionTimes);
    double variance = calculate_variance(executionTimes, mean);
    double sd = sqrt(variance);
    std::cout << "Mean: " << mean << ", Variance: " << variance << ", Standard Deviation: " << sd << std::endl;

}

int main(int argc, char const *argv[]) {

    if (argc == 3) {
        int exponent = std::stoi(argv[1]);
        int repetition = std::stoi(argv[2]);
        array_addition<int>(exponent, repetition);
        array_addition<float>(exponent, repetition);
    } else {
        std::cout << "2 arguments expected: exponent and repetition!" << std::endl;
    }
    
    return 0;
}
