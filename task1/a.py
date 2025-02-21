import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

# Load datasets
dataset_files = ['dataset_1.json', 'dataset_2.json', 'dataset_3.json']

datasets = []
for file in dataset_files:
    with open(file, 'r') as f:
        data = json.load(f)
        datasets.append([1 if x else 0 for x in data])  # Convert Boolean values to 1 (Heads) and 0 (Tails)

# Define Bayesian inference function
def bayesian_inference(coin_flips):
    N = len(coin_flips)  # Total number of flips
    M = sum(coin_flips)  # Number of heads
    
    # Define prior (uniform prior: P(p) = 1)
    p_values = np.linspace(0, 1, 1000)  # Range of p values
    likelihood = stats.binom.pmf(M, N, p_values)  # Binomial likelihood
    posterior = likelihood / np.trapz(likelihood, p_values)  # Normalize posterior
    
    # Compute expectation and variance
    expectation = np.trapz(p_values * posterior, p_values)
    variance = np.trapz((p_values**2) * posterior, p_values) - expectation**2

    return p_values, posterior, expectation, variance

# Compute Bayesian inference for all datasets
results = [bayesian_inference(dataset) for dataset in datasets]

# Plot results
plt.figure(figsize=(12, 4))
for i, (p_values, posterior, expectation, variance) in enumerate(results):
    plt.subplot(1, 3, i + 1)
    plt.plot(p_values, posterior, label=f"Dataset {i+1}")
    plt.axvline(expectation, color='r', linestyle="--", label=f"E[p]={expectation:.3f}")
    plt.fill_between(p_values, posterior, alpha=0.3)
    plt.xlabel("p (probability of heads)")
    plt.ylabel("Posterior Probability")
    plt.title(f"Posterior for Dataset {i+1}")
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("plot_a.png")

# Display expectation values and variances
df_results = pd.DataFrame({
    "Dataset": [1, 2, 3],
    "Expectation (E[p])": [res[2] for res in results],
    "Variance (Var[p])": [res[3] for res in results]
})

print(df_results)
