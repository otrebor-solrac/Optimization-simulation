# 🧩 Sudoku Solver using Genetic Algorithm

<p align="center">
  <img src="docs/image.png" alt="Sudoku Solver Output - Generation 999" width="480"/>
</p>

This directory contains an evolutionary **Genetic Algorithm (GA)** implementation engineered to solve standard **9x9 Sudoku** puzzles.

Sudoku is a well-known combinatorial Constraint Satisfaction Problem (CSP) that is NP-complete for generalized $N \times N$ grids. This project employs an evolutionary metaheuristic inspired by natural selection to explore candidate configurations and efficiently converge toward a valid solution.

---

## 📁 Directory Structure

* [`Sudoku-GA.ipynb`](file:///home/rc/workspace/Optimization-simulation/Sudoku-Genetic-Algorithm/Sudoku-GA.ipynb): Interactive Jupyter notebook containing the full implementation of [`SudokuGeneticAlgorithm`](file:///home/rc/workspace/Optimization-simulation/Sudoku-Genetic-Algorithm/Sudoku-GA.ipynb#L63), test puzzles, and live visualization.
* [`docs/`](file:///home/rc/workspace/Optimization-simulation/Sudoku-Genetic-Algorithm/docs): Visual assets and diagrams ([`image.png`](file:///home/rc/workspace/Optimization-simulation/Sudoku-Genetic-Algorithm/docs/image.png)).

---

## ⚙️ Algorithmic Mechanics

1. **Individual Representation:** Each chromosome represents a complete $9 \times 9$ grid. Clues provided in the initial puzzle are preserved as immutable coordinates in `fixed_positions` and are protected against crossover and mutation.
2. **Block-Aware Initialization:** Subgrids of size $3 \times 3$ are initialized by sampling permutations of the missing digits ($1$ through $9$). As a result, 3x3 block constraints are satisfied from the initial generation.
3. **Fitness Metric:** Quantifies unique digits across all 9 rows, 9 columns, and nine 3x3 blocks:
   $$\text{Maximum Fitness} = (9 \times 9)_{\text{rows}} + (9 \times 9)_{\text{cols}} + (9 \times 9)_{\text{blocks}} = 243$$
   Reaching a fitness score of 243 confirms a valid, conflict-free puzzle solution.
4. **Genetic Operators:**
   * **Crossover:** Row-based crossover (`_crossover_row`) and 3x3 block-based crossover (`_crossover_blocks`).
   * **Mutation:** Random cell mutation (`_mutate_simple`) and intelligent swap mutation within 3x3 blocks (`_mutate2`), preserving subgrid uniqueness.
5. **Convergence & Anti-Stagnation:**
   * Elitism: Preserves the top 5% fittest individuals.
   * Local Search: Greedy heuristic (`_local_improvement`) applied to elite candidates.
   * Plateau Detection: When fitness stalls for 5 generations, the algorithm dynamically increases the mutation rate, cleans duplicates, and filters out previously visited candidates using hash IDs (`history_best`).
6. **Live Visualization:** Real-time visual tracking of grid states and fitness progression using `matplotlib` and `IPython.display`.

---

## 🚀 How to Run

Ensure the required dependencies are installed:

```bash
pip install numpy matplotlib ipython jupyterlab
```

Launch Jupyter and open [`Sudoku-GA.ipynb`](file:///home/rc/workspace/Optimization-simulation/Sudoku-Genetic-Algorithm/Sudoku-GA.ipynb):

```bash
jupyter lab Sudoku-GA.ipynb
```
