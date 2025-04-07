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

def simulate_life_entropy(grid_size=(50, 50), steps=100, runs=10):
    all_entropies = np.zeros((runs, steps))

    for run in range(runs):
        grid = np.random.choice([0, 1], size=grid_size)
        for step in range(steps):
            H = compute_entropy(grid)
            all_entropies[run, step] = H
            grid = game_of_life_step(grid)

    return np.mean(all_entropies, axis=0)  

def plot_average_entropy(avg_entropy):
    plt.plot(avg_entropy, label="Mean Entropy")
    plt.xlabel("Time step")
    plt.ylabel("Entropy (Sh)")
    plt.title("Mean Entropy over Time in Conway's Game of Life")
    plt.grid(True)
    plt.legend()
    plt.show()

avg_entropy = simulate_life_entropy(grid_size=(50, 50), steps=200, runs=20)
plot_average_entropy(avg_entropy)
