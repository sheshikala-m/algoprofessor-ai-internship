"""
Day 03 — NumPy Operations & Numerical Analysis
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To perform fundamental NumPy operations including array creation,
statistical computations, reshaping, and random data generation.
"""

import numpy as np


def main():
    print("===== NumPy Numerical Analysis =====\n")

    # 1D Array Creation
    arr1 = np.array([10, 20, 30, 40, 50])
    print("1D Array:\n", arr1)

    # 2D Array Creation
    arr2 = np.array([[1, 2], [3, 4]])
    print("\n2D Array:\n", arr2)

    # Shape Analysis
    print("\nShape of 2D Array:", arr2.shape)

    # Statistical Metrics
    print("\nStatistical Analysis:")
    print("Mean:", np.mean(arr1))
    print("Standard Deviation:", np.std(arr1))
    print("Sum:", np.sum(arr1))

    # Reshaping Operation
    reshaped = arr1.reshape(5, 1)
    print("\nReshaped Array:\n", reshaped)

    # Random Data Generation
    random_arr = np.random.rand(3, 3)
    print("\nRandomly Generated Array:\n", random_arr)


if __name__ == "__main__":
    main()
