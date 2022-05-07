#import pandas as pd
import numpy as np

arr = np.arange(0, 49)
print(arr[48])
even_idx = arr[arr % 2 == 0]
odd_idx = arr[arr % 2 != 0]
arr[even_idx] = 0
arr[odd_idx] = 1
print(np.reshape(arr, (7, 7)))

print("Or !!!")

m = np.zeros((7, 7))
print(m)
m[::2, 1::2] = 1
# m[1::2, ::2] = 1
print(m)

diameters = np.array([1, 3, 5, 2, 4])
lengths = np.array([10, 20, 3, 10, 5])
pi = np.pi
print("Fresh")
print(np.array(np.power(diameters / 2, 2) * pi * lengths))
print(np.array(np.power(diameters / 2, 2) * lengths * np.pi))


def create_cartesian_product(vec1, vec2):
    return np.transpose([np.tile(vec1, len(vec2)), np.repeat(vec2, len(vec1))])


# print(create_cartesian_product([1, 2, 3], [4, 5, 6, 7]))

def closest_number_in_arr_to_n(arr, n):
    arr2 = abs(arr - n)
    #print("arr2:", arr2)
    target = min(np.unique(arr2))
    #print("Traget:", target)
    index_arr = arr2 != target
    #print(index_arr)
    arr[index_arr] = 0
    #print(max(np.unique(arr)))
    return max(np.unique(arr))


closest_number_in_arr_to_n(np.array([[1, 2, 3], [4, 5, 6]]), 4)
print("\n\n")
closest_number_in_arr_to_n(np.array([[1, 2, 3], [4, 5, 6], [21, 7, 5]]), 13)
print("\n\n")

def find_closest(a, n):
    a = np.array(a)
    print(a)
    return a[np.argmin(np.abs(a - n))]


# print(find_closest([1, 24, 12, 13, 14], 10))