def array_sum(arr, index=0):
    """
    STATE VARIABLES:
    - arr: The array (shared, read-only)
    - index: Current position (changes with each recursion)
    
    INVARIANT: Returns sum of arr[index:]
    """
    
    # BASE CASE: No more elements
    if index >= len(arr):
        return 0
    
    # RECURSIVE CASE:
    # Current element + sum of rest
    current_value = arr[index]
    rest_sum = array_sum(arr, index + 1)
    
    return current_value + rest_sum

print(array_sum([1,2,3], 0))
# Execution trace for [1, 2, 3]:
# array_sum([1,2,3], 0) = 1 + array_sum([1,2,3], 1)
#   array_sum([1,2,3], 1) = 2 + array_sum([1,2,3], 2)
#     array_sum([1,2,3], 2) = 3 + array_sum([1,2,3], 3)
#       array_sum([1,2,3], 3) = 0  (base case)
#     returns 3 + 0 = 3
#   returns 2 + 3 = 5
# returns 1 + 5 = 6
