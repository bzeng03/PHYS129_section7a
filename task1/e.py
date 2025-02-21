import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp
import pandas as pd

# Load datasets
dataset_files = ['dataset_1.json', 'dataset_2.json', 'dataset_3.json']

datasets = []
for file in dataset_files:
    with open(file, 'r') as f:
        data = json.load(f)
        datasets.append([1 if x else 0 for x in data])  # Convert Boolean values to 1 (Heads) and 0 (Tails)

# Bootstrapping Function
def bootstrap_statistics(data, sample_sizes, num_bootstrap=100):
    bootstrap_means = {size: [] for size in sample_sizes}
    bootstrap_variances = {size: [] for size in sample_sizes}

    for size in sample_sizes:
        for _ in range(num_bootstrap):
            sample = np.random.choice(data, size=size, replace=True)
            bootstrap_means[size].append(np.mean(sample))
            bootstrap_variances[size].append(np.var(sample))

    return bootstrap_means, bootstrap_variances

# Define sample sizes for bootstrapping
sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]

# Perform bootstrapping for all datasets
bootstrap_results = [bootstrap_statistics(dataset, sample_sizes) for dataset in datasets]

# Perform bootstrapping for all datasets
bootstrap_results = [bootstrap_statistics(dataset, sample_sizes) for dataset in datasets]

# Perform bootstrapping for all datasets
bootstrap_results = [bootstrap_statistics(dataset, sample_sizes) for dataset in datasets]

# Plot bootstrapped means histograms in a 9x3 grid (27 subplots)
fig, axes = plt.subplots(9, 3, figsize=(15, 30))
fig.suptitle("Bootstrapped Mean Distributions")

for i in range(3):  # Iterate over datasets
    for j, size in enumerate(sample_sizes):  # Iterate over sample sizes
        row = j  # Assign unique row for each sample size
        col = i  # Assign unique column for each dataset
        axes[row, col].hist(bootstrap_results[i][0][size], bins=20, alpha=0.6, label=f"Size {size}")
        axes[row, col].set_title(f"Dataset {i+1}, Sample {size}")
        axes[row, col].set_xlabel("Mean")
        axes[row, col].set_ylabel("Frequency")
        axes[row, col].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
plt.savefig("plot_e1.png")

# Collect bootstrap results into a DataFrame
bootstrap_data = []
for i, (bootstrap_means, bootstrap_variances) in enumerate(bootstrap_results):
    for size in sample_sizes:
        bootstrap_data.append({
            "Dataset": i + 1,
            "Sample Size": size,
            "Bootstrap Mean": np.mean(bootstrap_means[size]),
            "Bootstrap Variance": np.mean(bootstrap_variances[size])
        })

df_bootstrap_results = pd.DataFrame(bootstrap_data)

# Plot expectation and variance with respect to sample size
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle("Expectation and Variance vs. Sample Size")

for i in range(3):
    dataset_results = df_bootstrap_results[df_bootstrap_results["Dataset"] == i + 1]
    
    # Expectation plot
    axes[i, 0].plot(dataset_results["Sample Size"], dataset_results["Bootstrap Mean"], marker='o', linestyle='-')
    axes[i, 0].set_title(f"Dataset {i+1} - Expectation")
    axes[i, 0].set_xlabel("Sample Size")
    axes[i, 0].set_ylabel("Expectation")
    
    # Variance plot
    axes[i, 1].plot(dataset_results["Sample Size"], dataset_results["Bootstrap Variance"], marker='o', linestyle='-')
    axes[i, 1].set_title(f"Dataset {i+1} - Variance")
    axes[i, 1].set_xlabel("Sample Size")
    axes[i, 1].set_ylabel("Variance")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
plt.savefig("plot_e2.png")

# Display results
print(df_bootstrap_results)