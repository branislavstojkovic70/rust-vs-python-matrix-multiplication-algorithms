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
    println!("\n{} ({}x{}) - First 5x5 elements:", name, matrix.len(), matrix[0].len());
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
    
    // 8 paralelnih množenja - organizovanih u grupe od po 2
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
    
    let c11 = add_matrices(&c11_parts.0, &c11_parts.1);
    let c12 = add_matrices(&c12_parts.0, &c12_parts.1);
    let c21 = add_matrices(&c21_parts.0, &c21_parts.1);
    let c22 = add_matrices(&c22_parts.0, &c22_parts.1);
    
    let mut result = create_zero_matrix(n);
    set_submatrix(&mut result, &c11, 0, 0);
    set_submatrix(&mut result, &c12, 0, half);
    set_submatrix(&mut result, &c21, half, 0);
    set_submatrix(&mut result, &c22, half, half);
    
    result
}


fn strassen_multiply(a: &Matrix, b: &Matrix) -> Matrix {
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
    
    let m1 = strassen_multiply(&add_matrices(&a11, &a22), &add_matrices(&b11, &b22));
    let m2 = strassen_multiply(&add_matrices(&a21, &a22), &b11);
    let m3 = strassen_multiply(&a11, &subtract_matrices(&b12, &b22));
    let m4 = strassen_multiply(&a22, &subtract_matrices(&b21, &b11));
    let m5 = strassen_multiply(&add_matrices(&a11, &a12), &b22);
    let m6 = strassen_multiply(&subtract_matrices(&a21, &a11), &add_matrices(&b11, &b12));
    let m7 = strassen_multiply(&subtract_matrices(&a12, &a22), &add_matrices(&b21, &b22));
    
    let c11 = add_matrices(&subtract_matrices(&add_matrices(&m1, &m4), &m5), &m7);
    let c12 = add_matrices(&m3, &m5);
    let c21 = add_matrices(&m2, &m4);
    let c22 = add_matrices(&subtract_matrices(&add_matrices(&m1, &m3), &m2), &m6);
    
    let mut result = create_zero_matrix(n);
    set_submatrix(&mut result, &c11, 0, 0);
    set_submatrix(&mut result, &c12, 0, half);
    set_submatrix(&mut result, &c21, half, 0);
    set_submatrix(&mut result, &c22, half, half);
    
    result
}


fn strassen_parallel(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    
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
    
    let (m1_m2, m3_m4) = rayon::join(
        || rayon::join(
            || strassen_parallel(&add_matrices(&a11, &a22), &add_matrices(&b11, &b22)),
            || strassen_parallel(&add_matrices(&a21, &a22), &b11)
        ),
        || rayon::join(
            || strassen_parallel(&a11, &subtract_matrices(&b12, &b22)),
            || strassen_parallel(&a22, &subtract_matrices(&b21, &b11))
        )
    );
    
    let (m5_m6, m7) = rayon::join(
        || rayon::join(
            || strassen_parallel(&add_matrices(&a11, &a12), &b22),
            || strassen_parallel(&subtract_matrices(&a21, &a11), &add_matrices(&b11, &b12))
        ),
        || strassen_parallel(&subtract_matrices(&a12, &a22), &add_matrices(&b21, &b22))
    );
    
    let (m1, m2) = m1_m2;
    let (m3, m4) = m3_m4;
    let (m5, m6) = m5_m6;
    
    let c11 = add_matrices(&subtract_matrices(&add_matrices(&m1, &m4), &m5), &m7);
    let c12 = add_matrices(&m3, &m5);
    let c21 = add_matrices(&m2, &m4);
    let c22 = add_matrices(&subtract_matrices(&add_matrices(&m1, &m3), &m2), &m6);
    
    let mut result = create_zero_matrix(n);
    set_submatrix(&mut result, &c11, 0, 0);
    set_submatrix(&mut result, &c12, 0, half);
    set_submatrix(&mut result, &c21, half, 0);
    set_submatrix(&mut result, &c22, half, half);
    
    result
}


fn get_memory_usage() -> u64 {
    let mut sys = System::new_all();
    sys.refresh_all();
    sys.used_memory()
}

fn benchmark_algorithm<F>(name: &str, a: &Matrix, b: &Matrix, algorithm: F) 
where
    F: Fn(&Matrix, &Matrix) -> Matrix,
{
    println!("Testing: {}", name);
    
    let mem_before = get_memory_usage();
    let start = Instant::now();
    
    let result = algorithm(a, b);
    
    let duration = start.elapsed();
    let mem_after = get_memory_usage();
    let mem_used = if mem_after > mem_before { 
        mem_after - mem_before 
    } else { 
        0 
    };
    
    println!("Time to execute: {:.3} sec", duration.as_secs_f64());
    println!("Memory: ~{} MB", mem_used / 1_000_000);
    
    print_matrix_sample(&result, "Result");
}


fn main() {
    println!("\n Rust Implementation - Main ");
    
    let sizes = vec![128, 256, 512];
    
    for &size in &sizes {
        let a = create_random_matrix(size);
        let b = create_random_matrix(size);
        benchmark_algorithm("1. Iterative Algorithm", &a, &b, iterative_multiply);
        
        if size >= 256 {
            benchmark_algorithm("2. Divide & Conquer (sequential)", &a, &b, divide_conquer_multiply);
            benchmark_algorithm("3. Divide & Conquer (parralel)", &a, &b, divide_conquer_parallel);
            benchmark_algorithm("4. Strassen (sequential)", &a, &b, strassen_multiply);
            benchmark_algorithm("5. Strassen (parralel)", &a, &b, strassen_parallel);
        }
        
        println!("\n");
    }
    
}