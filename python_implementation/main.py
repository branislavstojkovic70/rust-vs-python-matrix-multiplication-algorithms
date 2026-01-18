import random
import time
from multiprocessing import Pool, cpu_count

def create_random_matrix(n: int) -> list:
    """Create a random n x n matrix"""
    return [[random.uniform(0, 10) for _ in range(n)] for _ in range(n)]

def create_zero_matrix(n: int) -> list:
    """Create a zero n x n matrix"""
    return [[0.0 for _ in range(n)] for _ in range(n)]

def print_matrix_sample(matrix: list, name: str):
    """Print the first 5x5 elements of a matrix"""
    n = len(matrix)
    print(f"\n{name} ({n}x{n}) - First 5x5 elements:")
    for i in range(min(5, n)):
        for j in range(min(5, n)):
            print(f"{matrix[i][j]:8.2f} ", end="")
        print()

def add_matrices(a: list, b: list) -> list:
    """Add two matrices"""
    n = len(a)
    result = create_zero_matrix(n)
    for i in range(n):
        for j in range(n):
            result[i][j] = a[i][j] + b[i][j]
    return result

def subtract_matrices(a: list, b: list) -> list:
    """Subtract two matrices"""
    n = len(a)
    result = create_zero_matrix(n)
    for i in range(n):
        for j in range(n):
            result[i][j] = a[i][j] - b[i][j]
    return result

def get_submatrix(matrix: list, row_start: int, col_start: int, size: int) -> list:
    """Extract a submatrix"""
    return [row[col_start:col_start + size] for row in matrix[row_start:row_start + size]]

def set_submatrix(result: list, submatrix: list, row_start: int, col_start: int):
    """Insert a submatrix into a larger matrix"""
    size = len(submatrix)
    for i in range(size):
        for j in range(size):
            result[row_start + i][col_start + j] = submatrix[i][j]

def iterative_multiply(a: list, b: list) -> list:
    """Standard iterative matrix multiplication O(n^3)"""
    n = len(a)
    result = create_zero_matrix(n)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]

    return result

def divide_conquer_multiply(a: list, b: list) -> list:
    """Divide & Conquer matrix multiplication (sequential)"""
    n = len(a)

    if n <= 64:
        return iterative_multiply(a, b)

    half = n // 2

    a11 = get_submatrix(a, 0, 0, half)
    a12 = get_submatrix(a, 0, half, half)
    a21 = get_submatrix(a, half, 0, half)
    a22 = get_submatrix(a, half, half, half)

    b11 = get_submatrix(b, 0, 0, half)
    b12 = get_submatrix(b, 0, half, half)
    b21 = get_submatrix(b, half, 0, half)
    b22 = get_submatrix(b, half, half, half)

    c11 = add_matrices(
        divide_conquer_multiply(a11, b11),
        divide_conquer_multiply(a12, b21)
    )
    c12 = add_matrices(
        divide_conquer_multiply(a11, b12),
        divide_conquer_multiply(a12, b22)
    )
    c21 = add_matrices(
        divide_conquer_multiply(a21, b11),
        divide_conquer_multiply(a22, b21)
    )
    c22 = add_matrices(
        divide_conquer_multiply(a21, b12),
        divide_conquer_multiply(a22, b22)
    )

    result = create_zero_matrix(n)
    set_submatrix(result, c11, 0, 0)
    set_submatrix(result, c12, 0, half)
    set_submatrix(result, c21, half, 0)
    set_submatrix(result, c22, half, half)

    return result

def _multiply_task(args):
    a, b = args
    if len(a) <= 64:
        return iterative_multiply(a, b)
    return divide_conquer_multiply(a, b)

def divide_conquer_parallel(a: list, b: list) -> list:
    """Divide & Conquer matrix multiplication (parallel)"""
    n = len(a)

    if n <= 128:
        return iterative_multiply(a, b)

    half = n // 2

    a11 = get_submatrix(a, 0, 0, half)
    a12 = get_submatrix(a, 0, half, half)
    a21 = get_submatrix(a, half, 0, half)
    a22 = get_submatrix(a, half, half, half)

    b11 = get_submatrix(b, 0, 0, half)
    b12 = get_submatrix(b, 0, half, half)
    b21 = get_submatrix(b, half, 0, half)
    b22 = get_submatrix(b, half, half, half)

    tasks = [
        (a11, b11), (a12, b21),
        (a11, b12), (a12, b22),
        (a21, b11), (a22, b21),
        (a21, b12), (a22, b22),
    ]

    with Pool(processes=min(8, cpu_count())) as pool:
        results = pool.map(_multiply_task, tasks)

    c11 = add_matrices(results[0], results[1])
    c12 = add_matrices(results[2], results[3])
    c21 = add_matrices(results[4], results[5])
    c22 = add_matrices(results[6], results[7])

    result = create_zero_matrix(n)
    set_submatrix(result, c11, 0, 0)
    set_submatrix(result, c12, 0, half)
    set_submatrix(result, c21, half, 0)
    set_submatrix(result, c22, half, half)

    return result

def strassen_multiply(a: list, b: list) -> list:
    """Strassen matrix multiplication (sequential)"""
    n = len(a)

    if n <= 64:
        return iterative_multiply(a, b)

    half = n // 2

    a11 = get_submatrix(a, 0, 0, half)
    a12 = get_submatrix(a, 0, half, half)
    a21 = get_submatrix(a, half, 0, half)
    a22 = get_submatrix(a, half, half, half)

    b11 = get_submatrix(b, 0, 0, half)
    b12 = get_submatrix(b, 0, half, half)
    b21 = get_submatrix(b, half, 0, half)
    b22 = get_submatrix(b, half, half, half)

    m1 = strassen_multiply(add_matrices(a11, a22), add_matrices(b11, b22))
    m2 = strassen_multiply(add_matrices(a21, a22), b11)
    m3 = strassen_multiply(a11, subtract_matrices(b12, b22))
    m4 = strassen_multiply(a22, subtract_matrices(b21, b11))
    m5 = strassen_multiply(add_matrices(a11, a12), b22)
    m6 = strassen_multiply(subtract_matrices(a21, a11), add_matrices(b11, b12))
    m7 = strassen_multiply(subtract_matrices(a12, a22), add_matrices(b21, b22))

    c11 = add_matrices(subtract_matrices(add_matrices(m1, m4), m5), m7)
    c12 = add_matrices(m3, m5)
    c21 = add_matrices(m2, m4)
    c22 = add_matrices(subtract_matrices(add_matrices(m1, m3), m2), m6)

    result = create_zero_matrix(n)
    set_submatrix(result, c11, 0, 0)
    set_submatrix(result, c12, 0, half)
    set_submatrix(result, c21, half, 0)
    set_submatrix(result, c22, half, half)

    return result
