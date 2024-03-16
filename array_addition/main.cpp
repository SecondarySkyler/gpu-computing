#include <iostream>
#include <fstream>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <numeric>

void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

int main(int argc, char const *argv[]) {

    if (argc == 3) {
        int exponent = std::stoi(argv[1]);
        int repetition = std::stoi(argv[2]);
        int arraySize = pow(2, exponent);
        std::vector<int> executionTimes;

        for (int i = 0; i < repetition; i++) {
            int* array_1 = new int[arraySize];
            int* array_2 = new int[arraySize];
            int* result = new int[arraySize];

            // Fill array_1 and array_2 with random numbers in range [0 - 100]
            for (int i = 0; i < arraySize; i++) {
                array_1[i] = (rand() % 100);
                array_2[i] = (rand() % 100);
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
        std::cout << "Times: ";
        for (auto times : executionTimes) {
            std::cout << times << ", ";
        }
        std::endl (std::cout);

        double mean = std::accumulate(executionTimes.begin(), executionTimes.end(), 0) / repetition;
        std::cout << "Mean: " << mean << std::endl;

        double var;
        for (auto times : executionTimes) {
            double aux = times - mean;
            var += pow(aux, 2);
        }

        var /= repetition;
        std::cout << "Variance: " << var << std::endl;

        double sd = sqrt(var);
        std::cout << "Standard Deviation: " << sd << std::endl;

    } else {
        std::cout << "2 arguments expected: exponent and repetition!" << std::endl;
    }

    double vm, rss;
    process_mem_usage(vm, rss);
    std::cout << "VM: " << vm << " RSS: " << rss << std::endl;
    
    return 0;
}
