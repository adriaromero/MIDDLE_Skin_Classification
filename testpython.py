#!/usr/bin/env python

import random   # Used for creating initial random arrays
import math     # Designate an "infinity"/sentinel value
import copy     # Create separate copies of the same array (list)
import timeit   # Measure execution time for functions

# infinity = math.inf
infinity = float("inf")
# number = 32767


# print the array to terminal --> [ .... ]
def print_array(cur_array):
    print("[ ", end="")
    for i in cur_array:
        print(i, end=" ")
    print("]")


# return a randomly generated array of specified length
def build_array(array_len):
    # 32767
    rand_array = []
    i = 0
    # rand_array.append('z')
    for i in range(array_len):
        cur_num = random.randint(0, 32767)
        rand_array.append(cur_num)
    return rand_array

# sort an array using insertion sort
def insertion_sort(cur_array):
    array_length = len(cur_array)
    # initialization
    for j in range(1, array_length):    # j = 2
        key = cur_array[j]
        i = j - 1
        while i >= 0 and cur_array[i] > key:
            cur_array[i+1] = cur_array[i]
            i -= 1
        cur_array[i+1] = key


# merge arrays used in merge sort
def merge_arrays(cur_array, first, div, last):
    n1 = div - first + 1
    n2 = last - div
    left_array = [None]*n1
    right_array = [None]*n2
    left_array.append(infinity)
    right_array.append(infinity)

    for i in range(n1):
        left_array[i] = cur_array[first + i]
    for j in range(n2):
        right_array[j] = cur_array[(div + 1) + j]  # need to check on this +1

    i, j = 0, 0
    for k in range(first, last + 1):  # need to check this +1
        if left_array[i] <= right_array[j]:
            cur_array[k] = left_array[i]
            i += 1
        else:
            cur_array[k] = right_array[j]
            j += 1

# sort an array using merge sort
def merge_sort(cur_array, first, last):
    if first < last:
        div = math.floor((first + last) / 2)
        merge_sort(cur_array, first, div)   # merge sort first half
        merge_sort(cur_array, div+1, last)  # merge sort second half
        merge_arrays(cur_array, first, div, last)


# partition used in quick sort
def partition(cur_array, first, last):
    value = cur_array[last]
    i = first - 1
    for j in range(first, last):
        if cur_array[j] <= value:
            i += 1
            # swap two array values
            temp = cur_array[i]
            cur_array[i] = cur_array[j]
            cur_array[j] = temp

    # swap two array values
    temp = cur_array[i+1]
    cur_array[i+1] = cur_array[last]
    cur_array[last] = temp
    i += 1
    return i


# sort an array using quick sort
def quick_sort(cur_array, first, last):
    if first < last:
        pivot = partition(cur_array, first, last)
        quick_sort(cur_array, first, pivot-1)
        quick_sort(cur_array, pivot+1, last)


# wraps function with keywords
def function_wrapper(func, *args, **kwards):
    # this function was adopted from http://pythoncentral.io/time-a-python-function/
    def wrapped():
        return func(*args, **kwards)
    return wrapped


def main():

    # -------------------------------------------------------------------------------------- set-up/initialization
    # ------ set the length of the array
    lenth_of_array = 8000
    last_index_of_array = lenth_of_array - 1

    # ------ Create three separate, identical, arrays to sort
    array_to_sort = build_array(lenth_of_array)             # original array
    insertion_sort_array = copy.deepcopy(array_to_sort)     # array copy for insertion sort
    merge_sort_array = copy.deepcopy(array_to_sort)         # array copy for merge sort
    quick_sort_array = copy.deepcopy(array_to_sort)         # array copy for quick sort

    # -------------------------------------------------------------------------------------- sort using each method
    # ------ insertion-sort
    insertion_wrapped = function_wrapper(insertion_sort, insertion_sort_array)
    time_insert_sort = timeit.timeit(insertion_wrapped, number=1)
    # insertion_sort(insertion_sort_array) # individual function call

    # ------ merge-sort
    merge_sort_wrapped = function_wrapper(merge_sort, merge_sort_array, 0, last_index_of_array)
    time_merge_sort = timeit.timeit(merge_sort_wrapped, number=1)
    # merge_sort(merge_sort_array, 0, last_index_of_array) # individual function call

    # ------ quick-sort
    quick_sort_wrapped = function_wrapper(quick_sort, quick_sort_array, 0, last_index_of_array)
    time_quick_sort = timeit.timeit(quick_sort_wrapped, number=1)
    # quick_sort(quick_sort_array, 0, last_index_of_array) # individual function call

    # -------------------------------------------------------------------------------------- print results
    print(time_insert_sort)
    print(time_merge_sort)
    print(time_quick_sort)



if __name__ == "__main__":
    main()
