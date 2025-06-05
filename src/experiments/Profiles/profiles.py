"""
Code to reproduce R(m) profiles in Fig 4 in paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# ------------------------------
# Function Definitions
# ------------------------------

def generate_data(n, m, s_params):
    """
    Generates data based on the provided parameters and computes m0, m1, m2, m3.

    Parameters:
    - n (int): Number of samples.
    - m (int): Number of chains.
    - s_params (dict): Dictionary containing s0, s1, s2, s3.

    Returns:
    - dict: Contains m0, m1, m2, m3 arrays and skewness values.
    """
    s0, s1, s2, s3 = s_params['s0'], s_params['s1'], s_params['s2'], s_params['s3']
    size = (n, m)

    # Data generation
    x0 = np.random.beta(s0, 1, size=size)
    x1 = np.random.gamma(s1, 1, size=size)
    x2 = np.random.binomial(1, s2, size=size)
    x3 = np.random.poisson(s3, size=size)

    # Initialize m arrays
    m0, m1, m2, m3 = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m)

    # Print skewness
    print(f"Skewness of x0 (Beta({s0}, 1)):", skew(x0.reshape(-1)))
    print(f"Skewness of x1 (Gamma({s1}, 1)):", skew(x1.reshape(-1)))
    print(f"Skewness of x2 (Binomial({s2})):", skew(x2.reshape(-1)))
    print(f"Skewness of x3 (Poisson({s3})):", skew(x3.reshape(-1)))
    print("-" * 50)

    # Define the function
    def f(x, i):
        x = x.copy()  # To avoid modifying the original data
        np.ceil(x)
        x[x >= 1000] = 1000
        return x[:, :i].max(1).mean() / x.mean()

    # Compute m0, m1, m2, m3
    for i in range(1, m):
        m0[i] = f(x0, i)
        m1[i] = f(x1, i)
        m2[i] = f(x2, i)
        m3[i] = f(x3, i)

    return {
        'm0': m0,
        'm1': m1,
        'm2': m2,
        'm3': m3
    }

def plot_data(ax, data, s_params, title, font_size, title_font_size):
    """
    Plots the data on the given axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to plot on.
    - data (dict): Contains m0, m1, m2, m3 arrays.
    - s_params (dict): Dictionary containing s0, s1, s2, s3.
    - title (str): Title of the subplot.
    - font_size (int): General font size.
    - title_font_size (int): Title font size.
    """
    ax.plot(data['m2'], label=f"Bern({s_params['s2']})", lw=2)
    ax.plot(data['m3'], label=f"Poiss({s_params['s3']})", lw=2)
    ax.plot(data['m0'], label=f"Beta({s_params['s0']}, 1)", lw=2)
    ax.plot(data['m1'], label=f"Gamma({s_params['s1']}, 1)", lw=2)

    # Set title with separate font size
    ax.set_title(title, fontsize=title_font_size)

    # Set labels with general font size
    ax.set_xlabel("# chains (m)", fontsize=font_size)
    ax.set_ylabel("R(m)", fontsize=font_size)

    # Set legend with general font size
    legend = ax.legend(loc='lower right', fontsize=font_size)

    # Set tick parameters with general font size
    ax.tick_params(axis='both', which='major', labelsize=font_size)

# ------------------------------
# Main Execution
# ------------------------------

def main():
    # Parameters for both plots
    n = 10000
    m = 1000

    # Font size settings
    font_size = 15       # General font size (XX)
    title_font_size = 15 # Title font size

    # Define parameters for both datasets
    plot_params = [
        {
            's_params': {'s0': 0.35, 's1': 4, 's2': 0.275, 's3': 1},
            'title': "Distributions with Skewness ≈ 1",
            'filename': "profiles_skew1.png"
        },
        {
            's_params': {'s0': 0.01, 's1': 0.04, 's2': 0.01, 's3': 0.01},
            'title': "Distributions with Skewness ≈ 10",
            'filename': "profiles_skew10.png"
        }
    ]

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # Adjust figsize as needed

    # Loop through each subplot and generate data and plot
    for ax, params in zip(axes, plot_params):
        data = generate_data(n, m, params['s_params'])
        plot_data(ax, data, params['s_params'], params['title'], font_size, title_font_size)

    # Adjust layout and save the combined figure
    plt.tight_layout()
    combined_filename = "combined_profiles.png"
    plt.savefig(combined_filename, bbox_inches="tight")
    plt.show()
    print(f"Combined figure saved as '{combined_filename}'.")

if __name__ == "__main__":
    main()