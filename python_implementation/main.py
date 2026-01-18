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