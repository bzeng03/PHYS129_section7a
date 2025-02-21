import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

# Stirling’s Approximation Validation
N_values = np.arange(1, 11)
log_factorial_exact = np.log(sp.gamma(N_values + 1))
stirling_approx = N_values * np.log(N_values) - N_values + 0.5 * np.log(2 * np.pi * N_values)
difference = log_factorial_exact - stirling_approx

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(N_values, log_factorial_exact, label="Exact log(n!)", color="blue")
plt.plot(N_values, stirling_approx, label="Stirling Approximation", linestyle="dashed", color="red")
plt.xlabel("N")
plt.ylabel("log(n!)")
plt.title("Comparison of Exact log(n!) and Stirling's Approximation")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(N_values, difference, marker="o", linestyle="dashed", color="green")
plt.xlabel("N")
plt.ylabel("Difference")
plt.title("Difference between Exact and Stirling's Approximation")

plt.tight_layout()
plt.show()
plt.savefig("plot_b.png")