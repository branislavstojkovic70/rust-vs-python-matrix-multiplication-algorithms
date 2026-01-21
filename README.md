# Matrix Multiplication: Rust vs Python

This repository contains implementations and an experimental comparison of multiple matrix multiplication algorithms written in **Rust** and **Python**.
The project was developed as part of a university seminar assignment focused on performance analysis, parallelization models, and resource usage.

The primary goal is to evaluate how different programming languages behave when implementing identical algorithms under the same experimental conditions.

---

## Implemented Algorithms

The following matrix multiplication algorithms are implemented in both languages:

1. **Iterative (Naive) Algorithm**

   * Time complexity: O(n³)
   * Serves as a baseline reference implementation

2. **Divide-and-Conquer Algorithm**

   * Recursive matrix decomposition into four submatrices
   * Same theoretical complexity as the iterative algorithm: O(n³)
   * Implemented in:

     * Sequential version
     * Parallel version

3. **Strassen’s Algorithm**

   * Reduces the number of recursive multiplications from 8 to 7 per recursion level
   * Time complexity: O(n^2.81)
   * Implemented in:

     * Sequential version
     * Parallel version

For recursive algorithms, a cutoff threshold is used to switch to the iterative algorithm for smaller matrix sizes.

---

## Project Structure

```text
.
├── rust_implementation/
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── src/
│       └── main.rs
│
├── python_implementation/
│   └── main.py
│
├── Rust_vs_Python_Mnozenje_Matrica_Seminarski.docx
├── Branislav_Stojkovic_E2_64_2025_Seminarski_Rust_i_Python.pdf
└── README.md
```

---

## Rust Implementation

* Language: Rust
* Compilation mode: `--release`
* Parallelization: `rayon` crate (`rayon::join`)
* Matrix representation: `Vec<Vec<f64>>`
* Memory usage measurement: `sysinfo` crate

Rust enables safe and efficient parallel execution through its ownership and borrowing model, eliminating data races at compile time. Parallel recursive calls are executed using a work-stealing thread pool provided by Rayon.

---

## Python Implementation

* Language: Python 3
* Parallelization: `multiprocessing` module
* Matrix representation: `list[list[float]]`
* Memory usage measurement: `psutil`

Due to the Global Interpreter Lock (GIL), parallel execution is achieved using multiple processes instead of threads. This introduces additional overhead related to process creation, inter-process communication, and memory duplication.

---

## Experimental Setup

* All benchmarks were executed on the same machine
* Matrix sizes:

  * 128 × 128
  * 256 × 256
  * 512 × 512
* Rust binaries were compiled in release mode
* Python code was executed using the standard interpreter

Execution time and memory usage were measured for each algorithm and configuration.

---

## Key Observations

* Rust significantly outperforms Python in all tested configurations
* Parallelization has a greater impact on performance than the choice of algorithm
* Rust’s parallel implementations scale efficiently with low overhead
* Python benefits from parallel execution, but performance gains are limited by multiprocessing overhead
* Strassen’s algorithm requires more memory due to additional temporary matrices

---

## Requirements

### Rust

* Rust toolchain (stable)
* Required crates:

  * `rayon`
  * `sysinfo`
  * `rand`

### Python

* Python 3.x
* Required packages:

  * `psutil`

Install Python dependencies with:

```bash
pip install psutil
```

---

## Running the Implementations

### Rust

```bash
cd rust_implementation
cargo run --release
```

### Python

```bash
cd python_implementation
python3 main.py
```

---

## Possible Extensions

* SIMD optimizations in Rust
* Comparison with NumPy / BLAS-based implementations
* GPU acceleration (CUDA, OpenCL)
* Optimized matrix storage using flat arrays for better cache locality

---

## Author

Branislav Stojković E2 64/2025<br>
Faculty of Technical Sciences<br>
University of Novi Sad

---
