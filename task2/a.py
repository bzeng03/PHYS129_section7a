import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.integrate import quad
from math import exp, sqrt, pi

# Load datasets
import json

with open('Vacuum_decay_dataset.json', 'r') as f:
    vacuum_data = json.load(f)

with open('Cavity_decay_dataset.json', 'r') as f:
    cavity_data = json.load(f)

# Subtract minimum value from each dataset
min_vacuum = min(vacuum_data)
min_cavity = min(cavity_data)
vacuum_data = [x - min_vacuum for x in vacuum_data]
cavity_data = [x - min_cavity for x in cavity_data]

# Define fitting functions
def exp_decay(x, lambda_):
    return (1 / lambda_) * np.exp(-x / lambda_)

def combined_decay(x, lmd, sgm, mu, normal_height):
    exponential = np.exp(-x/lmd)/lmd
    normal = np.exp(-(x-mu)**2/(2*sgm**2)) / (np.sqrt(2*np.pi) * sgm)
    return (1-normal_height)*exponential + normal_height*normal

# Bin the data for histogram
bins = np.linspace(0, 10, 50)
hist_vacuum, bin_edges_v = np.histogram(vacuum_data, bins=bins, density=True)
hist_cavity, bin_edges_c = np.histogram(cavity_data, bins=bins, density=True)

# Fit vacuum data
bin_centers_v = (bin_edges_v[:-1] + bin_edges_v[1:]) / 2
popt_v, pcov_v = curve_fit(exp_decay, bin_centers_v, hist_vacuum, p0=[1.0])

# Fit cavity data
bin_centers_c = (bin_edges_c[:-1] + bin_edges_c[1:]) / 2
popt_c, pcov_c = curve_fit(combined_decay, bin_centers_c, hist_cavity, p0=[1.25, 1, 6, 0.2])

# Compute Fisher Information Matrix
fisher_v = np.linalg.inv(pcov_v)
fisher_c = np.linalg.inv(pcov_c)

# Plot the histograms and fit curves
x_vals = np.linspace(0, 10, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vacuum decay
axes[0].hist(vacuum_data, bins=bins, density=True, alpha=0.6, color='b', label='Vacuum Data')
axes[0].plot(x_vals, exp_decay(x_vals, *popt_v), 'r-', label=f'Fit: λ={popt_v[0]:.2f}')
axes[0].set_title("Vacuum Decay")
axes[0].set_xlabel("Distance")
axes[0].set_ylabel("Density")
axes[0].legend()

# Cavity decay
axes[1].hist(cavity_data, bins=bins, density=True, alpha=0.6, color='g', label='Cavity Data')
axes[1].plot(x_vals, combined_decay(x_vals, *popt_c), 'r-', label=f'Fit: λ={popt_c[0]:.2f}, σ={popt_c[1]:.2f}, μ={popt_c[2]:.2f}, h={popt_c[3]:.2f}')
axes[1].set_title("Cavity Decay")
axes[1].set_xlabel("Distance")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.show()
plt.savefig("plot_a.png")

# Print Fisher Information Matrices
if __name__ == "__main__":
    print("Fisher Information Matrix for Vacuum Decay:")
    print(fisher_v)
    print("\nFisher Information Matrix for Cavity Decay:")
    print(fisher_c)