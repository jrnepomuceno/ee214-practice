import random
import matplotlib.pyplot as plt
import math
import sys

def main():

    n = 100
    # Generate random binary stream
    rand_bin = str(bin(random.randint(0, (2 ** n))))[2:]
    # pad zeroes at the start of string if needed until length n
    if len(rand_bin) < n:
        rand_bin = rand_bin.zfill(n)

    # data = binomial_rv(rand_bin, n)
    data = neg_binomial_rv(rand_bin, 34)


def combination(n, r):
    return (math.factorial(n)) // (math.factorial(r) * math.factorial(n - r))

def binomial_rv(rand_bin, n):
    p = round(list(rand_bin).count('1') / n, 2)
    result = [combination(n, x) * (p ** x) * ((1 - p) ** (n - x)) for x in range(0, n)]

    plt.hist(result, bins=20)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Data: p={p} n={n}")
    plt.show()

    return result

def neg_binomial_rv(rand_bin, k):
    
    rand_len = len(rand_bin)
    for index, item in enumerate(rand_bin):
        if item == '1': 
            k -= 1
            if k == 0: 
                rand_bin = rand_bin[:index+1]
                break

    if len(rand_bin) < rand_len:
        rand_bin = rand_bin.zfill(rand_len)

    # print(rand_bin, len(rand_bin), rand_bin.count('1'))
    result = [combination(x-1, k-1) * (p ** k) * ((1 - p) ** (x - k)) for x in range(0, rand_len)]

if __name__ == "__main__":
    main()