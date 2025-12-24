def factorial_acc(n, accumulator=1):
    """
    STATE:
    - n: remaining multiplications
    - accumulator: result so far
    
    TAIL RECURSIVE: Last operation is the recursive call
    """
    
    # BASE CASE
    if n == 0:
        return accumulator
    
    # RECURSIVE CASE: Update accumulator
    return factorial_acc(n - 1, accumulator * n)

print(factorial_acc(4, 1))
# factorial_acc(4, 1)   → factorial_acc(3, 4)
# factorial_acc(3, 4)   → factorial_acc(2, 12)
# factorial_acc(2, 12)  → factorial_acc(1, 24)
# factorial_acc(1, 24)  → factorial_acc(0, 24)
# factorial_acc(0, 24)  → 24
