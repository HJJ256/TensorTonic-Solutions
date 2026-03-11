import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_transpose = np.zeros((len(A[0]), len(A)))
    for i in range(A_transpose.shape[0]):
        for j in range(A_transpose.shape[1]):
            A_transpose[i,j]= A[j][i]
    return A_transpose
