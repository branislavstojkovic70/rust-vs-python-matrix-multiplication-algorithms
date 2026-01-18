use std::time::Instant;
use sysinfo::System;

type Matrix = Vec<Vec<f64>>;


fn create_random_matrix(n: usize) -> Matrix {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..n).map(|_| rng.gen_range(0.0..10.0)).collect())
        .collect()
}

fn create_zero_matrix(n: usize) -> Matrix {
    vec![vec![0.0; n]; n]
}

fn print_matrix_sample(matrix: &Matrix, name: &str) {
    println!("\n{} ({}x{}) - Prvi 5x5 elementi:", name, matrix.len(), matrix[0].len());
    for i in 0..matrix.len().min(5) {
        for j in 0..matrix[0].len().min(5) {
            print!("{:8.2} ", matrix[i][j]);
        }
        println!();
    }
}

fn add_matrices(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let mut result = create_zero_matrix(n);
    for i in 0..n {
        for j in 0..n {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

fn subtract_matrices(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let mut result = create_zero_matrix(n);
    for i in 0..n {
        for j in 0..n {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    result
}

fn main() {
    println!("Main start - Rust Implementation");
}
