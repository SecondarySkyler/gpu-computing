# Gpu Computing
Repository related to the Gpu Computing course held at the University of Trento for the A.Y. 2023/24 <br />
This repo contains the three deliverable required for the exam, which are listed below.

## Assignment 1
Required to implement a simple algorithm that transposes a no-symmetric matrix of size `N x N`, measuring the `Effective Bandwidth` of our implementation by using -00 –O1 –O2 –O3 options.
## Assignment 2
Required to implement a simple algorithm in `CUDA` that transposes a no-symmetric matrix of size `N x N`, measuring the `Effective Bandwidth` of our implementation.  
Furthermore, a comparison with results produced with the Assignment 1 was required.
## Project
The project required to design an efficient algorithm to transpose a sparse matrix. Specifically the matrix should be highly sparse, namely the number of zero element is more than 75% of the whole (n × n) elements. The implementation should emphasize:

- storage format for storing sparse matrices (for example, compressed sparse row);
- the implementation to perform the transposition;
- a comparison against vendors’ library (e.g., cuSPARSE);
- dataset for the benchmark (compare all the implementation presented by selecting at least 10 matrices from suite sparse matrix collection https://sparse.tamu.edu/);  

As usual, the metric to consider is the effective bandwidth.

