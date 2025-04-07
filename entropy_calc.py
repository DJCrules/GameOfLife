import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def shannon_entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def compute_entropy(grid):
    total_cells = grid.size
    live_cells = np.count_nonzero(grid)
    p = live_cells / total_cells
    return shannon_entropy(p)

def game_of_life_step(grid):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')
    birth = (grid == 0) & (neighbor_count == 3)
    survive = (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))
    return birth | survive

def simulate_life_entropy(grid_size=(50, 50), steps=100, runs=5):
    all_entropies = []

    for run in range(runs):
        grid = np.random.choice([0, 1], size=grid_size)
        entropies = []

        for _ in range(steps):
            H = compute_entropy(grid)
            entropies.append(H)
            grid = game_of_life_step(grid)

        all_entropies.append(entropies)

    return all_entropies

def plot_entropy(entropy_data):
    for i, entropies in enumerate(entropy_data):
        plt.plot(entropies, label=f"Run {i+1}")
    plt.xlabel("Time step")
    plt.ylabel("Entropy (Sh)")
    plt.title("Entropy over Time in Conway's Game of Life")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation and plot
entropy_results = simulate_life_entropy(grid_size=(100, 100), steps=300, runs=100)
plot_entropy(entropy_results)
