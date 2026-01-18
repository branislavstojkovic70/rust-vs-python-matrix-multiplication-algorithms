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

fn iterative_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let mut result = create_zero_matrix(n);
    
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn get_submatrix(matrix: &Matrix, row_start: usize, col_start: usize, size: usize) -> Matrix {
    matrix[row_start..row_start + size]
        .iter()
        .map(|row| row[col_start..col_start + size].to_vec())
        .collect()
}

fn set_submatrix(result: &mut Matrix, submatrix: &Matrix, row_start: usize, col_start: usize) {
    let size = submatrix.len();
    for i in 0..size {
        for j in 0..size {
            result[row_start + i][col_start + j] = submatrix[i][j];
        }
    }
}

fn divide_conquer_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    
    if n <= 64 {
        return iterative_multiply(a, b);
    }
    
    let half = n / 2;
    
    let a11 = get_submatrix(a, 0, 0, half);
    let a12 = get_submatrix(a, 0, half, half);
    let a21 = get_submatrix(a, half, 0, half);
    let a22 = get_submatrix(a, half, half, half);
    
    let b11 = get_submatrix(b, 0, 0, half);
    let b12 = get_submatrix(b, 0, half, half);
    let b21 = get_submatrix(b, half, 0, half);
    let b22 = get_submatrix(b, half, half, half);
    
    let c11 = add_matrices(
        &divide_conquer_multiply(&a11, &b11),
        &divide_conquer_multiply(&a12, &b21),
    );
    let c12 = add_matrices(
        &divide_conquer_multiply(&a11, &b12),
        &divide_conquer_multiply(&a12, &b22),
    );
    let c21 = add_matrices(
        &divide_conquer_multiply(&a21, &b11),
        &divide_conquer_multiply(&a22, &b21),
    );
    let c22 = add_matrices(
        &divide_conquer_multiply(&a21, &b12),
        &divide_conquer_multiply(&a22, &b22),
    );
    
    let mut result = create_zero_matrix(n);
    set_submatrix(&mut result, &c11, 0, 0);
    set_submatrix(&mut result, &c12, 0, half);
    set_submatrix(&mut result, &c21, half, 0);
    set_submatrix(&mut result, &c22, half, half);
    
    result
}


fn divide_conquer_parallel(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    
    // Bazni slučaj
    if n <= 128 {
        return iterative_multiply(a, b);
    }
    
    let half = n / 2;
    
    let a11 = get_submatrix(a, 0, 0, half);
    let a12 = get_submatrix(a, 0, half, half);
    let a21 = get_submatrix(a, half, 0, half);
    let a22 = get_submatrix(a, half, half, half);
    
    let b11 = get_submatrix(b, 0, 0, half);
    let b12 = get_submatrix(b, 0, half, half);
    let b21 = get_submatrix(b, half, 0, half);
    let b22 = get_submatrix(b, half, half, half);
    
    let (c11_parts, c12_parts) = rayon::join(
        || rayon::join(
            || divide_conquer_parallel(&a11, &b11),
            || divide_conquer_parallel(&a12, &b21)
        ),
        || rayon::join(
            || divide_conquer_parallel(&a11, &b12),
            || divide_conquer_parallel(&a12, &b22)
        )
    );
    
    let (c21_parts, c22_parts) = rayon::join(
        || rayon::join(
            || divide_conquer_parallel(&a21, &b11),
            || divide_conquer_parallel(&a22, &b21)
        ),
        || rayon::join(
            || divide_conquer_parallel(&a21, &b12),
            || divide_conquer_parallel(&a22, &b22)
        )
    );
    
    // Sabiranje parova rezultata
    let c11 = add_matrices(&c11_parts.0, &c11_parts.1);
    let c12 = add_matrices(&c12_parts.0, &c12_parts.1);
    let c21 = add_matrices(&c21_parts.0, &c21_parts.1);
    let c22 = add_matrices(&c22_parts.0, &c22_parts.1);
    
    // Kombinovanje
    let mut result = create_zero_matrix(n);
    set_submatrix(&mut result, &c11, 0, 0);
    set_submatrix(&mut result, &c12, 0, half);
    set_submatrix(&mut result, &c21, half, 0);
    set_submatrix(&mut result, &c22, half, half);
    
    result
}

fn main() {
    println!("Main start - Rust Implementation");
}
