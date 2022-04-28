import math

def exp(x):
    res = 0
    try:
        res = math.exp(x)
    except OverflowError:
        res = float('inf')

    return res

        
