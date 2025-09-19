import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, binom
import math

"""
PART 1
"""
def generate_biased_bitstream(length, p) -> str:
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1.")
    bitstream = map(lambda x: "1" if x < p else "0", [random.random() for _ in range(length)])
    return "".join(bitstream)

def combination(n, r):
    return (math.factorial(n)) // (math.factorial(r) * math.factorial(n-r))

def binomial_sim_rv(n, p, num_samples) -> list[int]:
    samples = [generate_biased_bitstream(n, p).count("1") for _ in range(0, num_samples)]
    return samples

def neg_binomial_sim_rv(k, p, num_samples) -> list[int]:
    samples = []
    for _ in range(0,num_samples):
        count = 0
        trials = 0
        while count < k:
            if random.random() < p:
                count += 1
            trials += 1
        samples.append(trials)

    return samples

def binomial_theo_rv(n, p):
    return binom.pmf(np.arange(0, n+1), n, p)

def neg_binomial_theo_rv(k, n, p):
    return [combination(x-1, k-1) * ((p ** k) * (1 - p) ** (x - k)) for x in range(k, n)]

"""
PART 2
"""
def exponential_sim_rv(rate, num_samples) -> list[np.float16]:
    uni_stream = [round(random.uniform(0, 1), 5) for _ in range(0, num_samples)]
    # using inverse of CDF
    exp_cdf_inverse_func = lambda y: round(-1*(np.log(1 - y)) / rate, 2)
    samples = list(map(exp_cdf_inverse_func, uni_stream))
    return samples

def gaussian_sim_rv(mu, sigma, num_samples):
    uni_stream = [round(random.uniform(0, 1), 5) for _ in range(0, num_samples)]
    # Getting inverse by using z-score (inverse norm)
    z_values = [norm.ppf(u) for u in uni_stream]
    converted_values = [sigma*z + mu for z in z_values]
    return converted_values

def exponential_theo_rv(r, n):
    return [r * (math.e ** (-1*r*x)) for x in range(0, 10)]

def gaussian_theo_rv(mu, sigma):
    return norm.pdf(np.linspace(-4, 4, 100), loc=mu, scale=sigma)

def main():

    samples = 1000
    n = 100

    """
    PART 1
    """
    # p = 0.3
    # sim_data_1 = binomial_sim_rv(n, p, samples)
    # th_data_1 = binomial_theo_rv(n, p)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle(f"Binomial RV Probability Distribution n={n} p={p}")
    # ax1.hist(sim_data_1, bins=20)
    # ax1.set_xlim(0, n)
    # ax1.set_title("Simulated")

    # ax2.bar(range(0, len(th_data_1)), th_data_1)
    # ax2.set_xlim(0, n)
    # ax2.set_title("Theoretical")
    # plt.tight_layout()

    # for index, n in enumerate([50, 75, 100]):
    #     plt.subplot(3, 1, index+1)
    #     plt.title(f"Binomial RV Distributions n={n}")
    #     for p in [0.3, 0.5, 0.7]:
    #         sim_data = binomial_sim_rv(n, p, samples)
    #         plt.hist(sim_data, label=f"p={p}", bins=100, alpha=0.7)
    #         plt.legend()
    # plt.tight_layout()

    # k = 15
    # p_2 = 0.3
    # sim_data_2 = neg_binomial_sim_rv(k, p_2, 1000)
    # th_data_2 = neg_binomial_theo_rv(k, n, p_2)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle(f"Negative Binomial RV Probability Distribution k={k} p={p_2}")
    # ax1.hist(sim_data_2, bins=20, density=True)
    # ax1.set_xlim(0, n)
    # ax1.set_title("Simulated")

    # ax2.bar(range(0, len(th_data_2)), th_data_2)
    # ax2.set_xlim(0, n)
    # ax2.set_title("Theoretical")
    # plt.tight_layout()

    # for index, k in enumerate([10, 15, 20]):
    #     plt.subplot(3, 1, index+1)
    #     plt.title(f"Negative Binomial RV Distributions k={k}")
    #     for p in [0.3, 0.5, 0.7]:
    #         sim_data = neg_binomial_sim_rv(k, p, samples)
    #         plt.hist(sim_data, label=f"p={p}", bins=100, alpha=0.7)
    #         plt.legend()
    # plt.tight_layout()

    # """
    # PART 2
    # """
    # r = 0.5
    # sim_data_3 = exponential_sim_rv(r, samples)
    # th_data_3 = exponential_theo_rv(r, samples)
    # plt.figure(1)
    # plt.suptitle(f"Exponential RV Probability Distribution lambda={r}")

    # plt.hist(sim_data_3, bins=n, density=True, label="Simulated")
    # plt.plot(th_data_3, color='red', label="Theoretical")
    # plt.legend()
    # plt.tight_layout()

    # for r in [0.1, 0.5, 1]:
    #     plt.title(f"Exponential RV Distributions")
    #     sim_data = exponential_sim_rv(r, samples)
    #     plt.hist(sim_data, label=f"r={r}", bins=100, alpha=0.5)
    #     plt.legend()
    # plt.tight_layout()

    # mu = 0
    # sigma = 1
    # sim_data_4 = gaussian_sim_rv(mu, sigma, 1000)
    # th_data_4 = gaussian_theo_rv(mu, sigma)
    # plt.figure(2)
    # plt.suptitle(f"Gaussian RV Probability Distribution mu={mu} sigma={sigma}")
    # plt.hist(sim_data_4, bins=n, density=True, label="Simulated")
    # plt.plot(np.linspace(-4, 4, 100), th_data_4, color='red', label="Theoretical")
    # plt.legend()
    # plt.tight_layout()

    # for sigma in [2, 1, 0.5]:
    #     plt.title(f"Gaussian RV Distributions")
    #     sim_data = gaussian_sim_rv(0, sigma, 1000)
    #     plt.hist(sim_data, label=f"sigma={sigma}", bins=100, alpha=0.5)
    #     plt.legend()
    # plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()