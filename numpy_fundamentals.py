import numpy as np

# 1D Array
a = np.array([1, 2, 3, 4, 5])
print("1D Array:", a)
print("1D Shape:", a.shape)

# 2D Array
b = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", b)
print("2D Shape:", b.shape)

# 3D Array
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D Array:\n", c)
print("3D Shape:", c.shape)

# Vectorized operation
print("Vectorized multiplication:", a * 2)

# Broadcasting
print("Broadcasting:", a + 10)

# Matrix multiplication
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print("Matrix Multiplication:\n", x @ y)

# Statistics

# NumPy fundamentals practical completed