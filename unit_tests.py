from bcolors import bcolors
from Strassen import *
import numpy as np

# Unit tests to check the correctness of the Strassen's Algorithm and the naive matrix multiplication code


# Test 1: Compare the results of the naive matrix multiplication and the Strassen's Algorithm on random 4x4 matrices (power of 2)
A = generate_random_matrix(4)
B = generate_random_matrix(4)

product_actual = np.matmul(A, B)

product_naive = matrix_mult(A, B)
product_strss = strassen(A, B)

if np.array_equal(product_actual, product_naive):
    print(bcolors.OKGREEN + "Test 1a: Naive MatMult (4x4) Passed" + bcolors.ENDC)
else:
    print(bcolors.FAIL + "Test 1a: Naive MatMult (4x4) Failed" + bcolors.ENDC)

if np.array_equal(product_actual, product_strss):
    print(bcolors.OKGREEN + "Test 1b: Strassen's Algorithm (4x4) Passed" + bcolors.ENDC + "\n")
else:
    print(bcolors.FAIL + "Test 1b: Strassen's Algorithm (4x4) Failed" + bcolors.ENDC + "\n")



# Test 2: Correctness for sizes not a power of 2, n = 9
    
A = generate_random_matrix(9)
B = generate_random_matrix(9)
product_actual = np.matmul(A, B)
product_naive = matrix_mult(A, B)
product_strss = strassen(A, B)

if np.array_equal(product_actual, product_naive):
    print(bcolors.OKGREEN + "Test 2a: Naive MatMult (9x9) Passed" + bcolors.ENDC)
else:
    print(bcolors.FAIL + "Test 2a: Naive MatMult (9x9) Failed" + bcolors.ENDC)

if np.array_equal(product_actual, product_strss):
    print(bcolors.OKGREEN + "Test 2b: Strassen's Algorithm (9x9) Passed" + bcolors.ENDC + "\n")
else:
    print(bcolors.FAIL + "Test 2b: Strassen's Algorithm (9x9) Failed" + bcolors.ENDC + "\n")


# Test 3: Correctness for sizes not a power of 2 and large, n = 150
A = generate_random_matrix(150)
B = generate_random_matrix(150)
product_actual = np.matmul(A, B)
product_naive = matrix_mult(A, B)
product_strss = strassen(A, B)

if np.array_equal(product_actual, product_naive):
    print(bcolors.OKGREEN + "Test 3a: Naive MatMult (150x150) Passed" + bcolors.ENDC)
else:
    print(bcolors.FAIL + "Test 3a: Naive MatMult (150x150) Failed" + bcolors.ENDC)

if np.array_equal(product_actual, product_strss):
    print(bcolors.OKGREEN + "Test 3b: Strassen's Algorithm (150x150) Passed" + bcolors.ENDC + "\n")
else:
    print(bcolors.FAIL + "Test 3b: Strassen's Algorithm (150x150) Failed" + bcolors.ENDC + "\n")
    
