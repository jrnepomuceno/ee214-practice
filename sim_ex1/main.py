import random
import matplotlib.pyplot as plt
import math
import numpy as np
import sys

def show_plot(x_range, data, p, name, x_label="Successes", y_label="", figure_num=1) -> None:

    plt.figure(figure_num)
    plt.bar(x_range, data)
    plt.xlabel(f"Number of {x_label}")
    plt.ylabel("Probability")
    plt.title(f"Probability Distribution of {name} p={p}")

def generate_biased_bitstream(length, p) -> str:
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1.")
    bitstream = map(lambda x: "1" if x > p else "0", [random.randint(0,1) for _ in range(length)])
    return "".join(bitstream)

def combination(n, r) -> float:
    return (math.factorial(n)) // (math.factorial(r) * math.factorial(n - r))

"""
PART 1
"""
def binomial_sim_rv(n, p, num_samples) -> list[int]:
    samples = [generate_biased_bitstream(n, p).count("1") for _ in range(0,num_samples)]
    return samples

def neg_binomial_sim_rv(k, n, p, num_samples) -> list[int]:
    samples = []
    for _ in range(0,num_samples):
        bits = generate_biased_bitstream(n, p)
        for i in range(1, n):
            if bits[:i].count("1") == k:
                samples.append(i)
                break

    return samples

def binomial_theo_rv(n, p=0.) -> list[int]:

    x_range = np.arange(0, n)
    result = [combination(n, x) * ((float(p) ** x) * ((1 - float(p)) ** (n - x))) for x in x_range]

    show_plot(x_range, result, p, name=f"Binomial RV: n={n}", x_label="Successes", figure_num=1)

    return result

def neg_binomial_theo_rv(k, p=0., num_trials=0):

    x_range = range(k, num_trials)
    result = [combination(x-1, k-1) * ((p ** k) * (1 - p) ** (x - k)) for x in x_range]
    show_plot(x_range, result, p, name=f"Negative Binomial RV: k={k}", x_label="Trials", figure_num=2)
    return result


"""
PART 2
"""

def exponential_sim_rv(rate, num_samples) -> list[np.float16]:
    uni_stream = [round(random.uniform(0, 1), 5) for _ in range(0, num_samples)]
    exp_cdf_inverse_func = lambda y: round(-1*(np.log(1 - y)) / rate, 2)
    samples = list(map(exp_cdf_inverse_func, uni_stream))
    return samples

def main():

    n = 100
    p = 0.3

    # data_1 = binomial_sim_rv(n, p, 1000)
    # plt.figure(1)
    # plt.hist(data_1, bins=n)

    # data_2 = neg_binomial_sim_rv(34, n, p, 1000)
    # # plt.figure(2)
    # plt.hist(data_2, bins=n)

    data_3 = exponential_sim_rv(0.5, 500)
    plt.hist(data_3, bins=n)

    # b_data, b_p = binomial_theo_rv(n, p, rand_bin=rand_bin)
    # negb_data, negb_p = neg_binomial_theo_rv(34, p, num_trials=100, rand_bin=rand_bin)

    plt.show()

if __name__ == "__main__":
    main()