import numpy as np
import math
import time

def generate_random_matrix(n):
    # Generate an n x n matrix with random values between 1 and 10
    return np.random.randint(0, 2, size=(n, n))

def split_matrix(a):
    # Split a matrix into 4 equal quadrants
    mid = a.shape[0] // 2
    return a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]

# Check if a number is a power of 2
def check_power_two(n):
    while (n % 2 == 0 and n > 1):
       n = n // 2
    return n == 1

# Multiply two matrices using the naive algorithm O(n^3)
def matrix_mult(matrix_a, matrix_b):
    n = matrix_a.shape[0]  # Use .shape[0] for consistency
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
    return result

# Find the next power of 2
# Example: nextPowerOf2(5) = 8 since 2^2 < 5 < 2^3
def nextPowerOf2(n):
    return 2 ** (math.ceil(math.log(n, 2)))

# Pad a matrix with zeroes to the next power of 2
def padding_zeroes(M)->np.ndarray:
    n = M.shape[0]
    M_padded_size = nextPowerOf2(n)
    return np.pad(M, ((0, M_padded_size - n), (0, M_padded_size - n)), mode='constant')

# Strassen's Algorithm: O(n^log7)
def strassen(matrix_a, matrix_b):
    n = matrix_a.shape[0]

    # Check if the matrix is a power of 2
    # If not, pad the matrix with zeroes
    if not check_power_two(n):
        matrix_a = padding_zeroes(matrix_a)
        matrix_b = padding_zeroes(matrix_b)
    
    if n == 2:
        return matrix_mult(matrix_a, matrix_b)
    
    A, B, C, D = split_matrix(matrix_a)
    E, F, G, H = split_matrix(matrix_b)

    # Strassen's Algorithm
    p1 = strassen(A, F - H)
    p2 = strassen(A + B, H)
    p3 = strassen(C + D, E)
    p4 = strassen(D, G - E)
    p5 = strassen(A + D, E + H)
    p6 = strassen(B - D, G + H)
    p7 = strassen(A - C, E + F)

    # Calculate the 4 quadrants of the final matrix
    top_left = p5 + p4 - p2 + p6
    top_right = p1 + p2
    bot_left = p3 + p4
    bot_right = p1 + p5 - p3 - p7

    # Combine the 4 quadrants into a single matrix
    new_matrix_top = np.hstack((top_left, top_right))
    new_matrix_bot = np.hstack((bot_left, bot_right))

    # Return the final matrix
    return (np.vstack((new_matrix_top, new_matrix_bot)))[:n,:n]

k = 10
n_values = [2**i for i in range(1,k+1)]
runtimes_strassen = k * [0]
runtimes_simple = k * [0]
# for i in range(len(n_values)):
#     print(i)
#     n = n_values[i]
#     m1 = generate_random_matrix(n)
#     m2 = generate_random_matrix(n)
#     start = time.time()
#     res = strassen(m1, m2)
#     end = time.time()
#     runtimes_strassen[i] = end - start
#     start = time.time()
#     res = matrix_mult(m1,m2)
#     end = time.time()
#     runtimes_simple[i] = end - start

#print("Naive runtimes are", runtimes_simple)
#print("Strassen Runtimes Are:", runtimes_strassen)

# part c

"""
def gen_graph(n, p):
    upper_triangle = np.triu(np.random.random((n, n)) < p, 1).astype(int)
    adjacency_matrix = upper_triangle + upper_triangle.T
    return adjacency_matrix

p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
size = 64
for p in p_values:
    G = gen_graph(size,p)
    res = strassen(G, strassen(G,G))
    output = np.trace(res)/6
    print(output)
"""
    
