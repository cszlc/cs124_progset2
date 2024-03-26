import numpy as np
import math
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# Helper functions like generate_random_matrix, check_power_two, nextPowerOf2, padding_zeroes, etc., remain the same
def generate_random_matrix(n):
    # Generate an n x n matrix with random values between 1 and 10
    return np.random.randint(0, 2, size=(n, n))

def split_matrix(a):
    mid = a.shape[0] // 2
    return a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]

def matrix_mult(matrix_a, matrix_b):
    n = matrix_a.shape[0] 
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
    return result

def pad_to_even(matrix):
    rows, cols = matrix.shape
    pad_rows = rows % 2 != 0
    pad_cols = cols % 2 != 0
    if pad_rows or pad_cols:
        padding = ((0, int(pad_rows)), (0, int(pad_cols)))
        matrix = np.pad(matrix, padding).astype(int)
    return matrix


def strassen(matrix_a, matrix_b):
    n = matrix_a.shape[0]
    
    matrix_a = pad_to_even(matrix_a)
    matrix_b = pad_to_even(matrix_b)
    
    if n <= 105:
        result = matrix_mult(matrix_a, matrix_b)
        return result[:n, :n]
    
    A, B, C, D = split_matrix(matrix_a)
    E, F, G, H = split_matrix(matrix_b)
    
    with ThreadPoolExecutor() as executor:
        p1_future = executor.submit(strassen, A, np.subtract(F, H))
        p2_future = executor.submit(strassen, np.add(A, B), H)
        p3_future = executor.submit(strassen, np.add(C, D), E)
        p4_future = executor.submit(strassen, D, np.subtract(G, E))
        p5_future = executor.submit(strassen, np.add(A, D), np.add(E, H))
        p6_future = executor.submit(strassen, np.subtract(B, D), np.add(G, H))
        p7_future = executor.submit(strassen, np.subtract(A, C), np.add(E, F)) 

        p1 = p1_future.result()
        p2 = p2_future.result()
        p3 = p3_future.result()
        p4 = p4_future.result()
        p5 = p5_future.result()
        p6 = p6_future.result()
        p7 = p7_future.result()

    top_left = p5 + p4 - p2 + p6
    top_right = p1 + p2
    bot_left = p3 + p4
    bot_right = p1 + p5 - p3 - p7

    # Construct the result matrix from the quadrants
    new_matrix_top = np.hstack((top_left, top_right))
    new_matrix_bot = np.hstack((bot_left, bot_right))
    full_result = np.vstack((new_matrix_top, new_matrix_bot))

    # Return the top-left submatrix that corresponds to the original size
    return full_result[:n, :n].astype(int)

# Read input matrices from a file
def read_matrices(filename, n):
    with open(filename, 'r') as file:
        data = file.read().splitlines()
    data = [int(float(val.strip())) for val in data]
    matrix_a = np.array(data[:n**2]).reshape(n, n)
    matrix_b = np.array(data[n**2:]).reshape(n, n)
    return matrix_a, matrix_b

# Output the diagonal elements of the matrix
def output_diagonal(matrix):
    for i in range(matrix.shape[0]):
        print(int(matrix[i, i]))

def main(flag, dimension, inputfile):
    n = int(dimension)
    matrix_a, matrix_b = read_matrices(inputfile, n)
    result = strassen(matrix_a, matrix_b)
    output_diagonal(result)

if __name__ == "__main__":
    # The program is called with the flag, the dimension of the matrices, and the filename of the input
    if len(sys.argv) != 4:
        print("Usage: python strassen.py flag dimension inputfile")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])