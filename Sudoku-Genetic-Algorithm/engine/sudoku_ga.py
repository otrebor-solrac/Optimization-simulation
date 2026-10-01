#!/usr/bin/env python3
import sys
import json
import uuid
import random
import threading
import select
from collections import deque
import numpy as np

GRID_SIZE = 9

# Puzzles from the original notebook
PUZZLES = {
    "easy": [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    "medium": [
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 0, 3],
        [0, 7, 4, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 2],
        [0, 8, 0, 0, 4, 0, 0, 1, 0],
        [6, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 7, 8, 0],
        [5, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0]
    ],
    "hard": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 8, 5],
        [0, 0, 1, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 7, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 1, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 7, 3],
        [0, 0, 2, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 9]
    ],
    "escargot": [
        [1, 0, 0, 0, 0, 7, 0, 9, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 8],
        [0, 0, 9, 6, 0, 0, 5, 0, 0],
        [0, 0, 5, 3, 0, 0, 9, 0, 0],
        [0, 1, 0, 0, 8, 0, 0, 0, 2],
        [6, 0, 0, 0, 0, 4, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 7, 0, 0, 0, 3, 0, 0]
    ]
}


def emit_puzzles_catalog():
    catalog = {}
    labels = {
        "easy": "EASY (30 clues)",
        "medium": "MEDIUM (24 clues)",
        "hard": "HARD (21 clues)",
        "escargot": "EVIL (AI Escargot)"
    }
    for key, matrix in PUZZLES.items():
        mat = np.array(matrix, dtype=int)
        fixed_mask = [1 if mat[r][c] != 0 else 0 for r in range(9) for c in range(9)]
        catalog[key] = {
            "label": labels.get(key, key.upper()),
            "grid": mat.flatten().tolist(),
            "fixed": fixed_mask,
            "clues": sum(fixed_mask)
        }
    sys.stdout.write(json.dumps({"type": "catalog", "puzzles": catalog}) + "\n")
    sys.stdout.flush()


class SudokuGeneticAlgorithm:
    def __init__(self,
                 pop_size=100,
                 max_generations=5000,
                 mutation_rate=0.05,
                 sudoku_puzzle=None):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.initial_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        if sudoku_puzzle is None:
            sudoku_puzzle = PUZZLES["default"]
        self.sudoku_puzzle = np.array(sudoku_puzzle, dtype=int)
        self.best_fitness = 0
        self.fixed_positions = {
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if self.sudoku_puzzle[r][c] != 0
        }
        self.fixed_rows = {r: set() for r in range(GRID_SIZE)}
        self.fixed_cols = {c: set() for c in range(GRID_SIZE)}
        for (r, c) in self.fixed_positions:
            val = int(self.sudoku_puzzle[r][c])
            self.fixed_rows[r].add(val)
            self.fixed_cols[c].add(val)

        self.non_fixed_positions = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) not in self.fixed_positions
        ]

        self.history_best = set()
        self.tabu_queue = deque(maxlen=300)
        self.tabu_set = set()
        # Matriz de Probabilidad (Feromonas). Dimensiones: 9x9, y 10 valores (el 0 no se usa, del 1 al 9).
        self.confidence_matrix = np.zeros((GRID_SIZE, GRID_SIZE, 10))

        # Identificar los 3 bloques con más pistas fijas (Bloques Ancla)
        block_clues = {}
        for br in range(3):
            for bc in range(3):
                cnt = sum(
                    1 for r in range(br * 3, (br + 1) * 3)
                    for c in range(bc * 3, (bc + 1) * 3)
                    if (r, c) in self.fixed_positions
                )
                block_clues[(br, bc)] = cnt
        self.anchor_blocks = sorted(block_clues.keys(), key=lambda b: block_clues[b], reverse=True)[:3]
        
        self.paused = False
        self.delay = 0.40  # 400ms between generations by default for clear visualization
        self.step_mode = False

    def _is_valid_assignment(self, r, c, val):
        # A cell (r, c) can never take a value that already exists as a fixed clue in row r or col c
        return (val not in self.fixed_rows[r]) and (val not in self.fixed_cols[c])

    def _add_tabu(self, grid):
        key = grid.tobytes()
        if len(self.tabu_queue) >= self.tabu_queue.maxlen:
            old = self.tabu_queue.popleft()
            self.tabu_set.discard(old)
        self.tabu_queue.append(key)
        self.tabu_set.add(key)

    def _is_tabu(self, grid):
        return grid.tobytes() in self.tabu_set

    def emit(self, data):
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    def _generate_population_blocks(self, total_pop):
        population = []
        for _ in range(total_pop):
            individual = np.copy(self.sudoku_puzzle)
            for block_row in range(3):
                for block_col in range(3):
                    r_start, c_start = block_row * 3, block_col * 3
                    block_coords = [
                        (r, c)
                        for r in range(r_start, r_start + 3)
                        for c in range(c_start, c_start + 3)
                    ]
                    fixed_vals = set()
                    non_fixed_coords = []
                    for r, c in block_coords:
                        val = individual[r][c]
                        if (r, c) in self.fixed_positions:
                            fixed_vals.add(val)
                        else:
                            non_fixed_coords.append((r, c))

                    available = list(set(range(1, 10)) - fixed_vals)
                    np.random.shuffle(available)

                    def solve_block(idx):
                        if idx == len(non_fixed_coords):
                            return True
                        r, c = non_fixed_coords[idx]
                        for i in range(len(available)):
                            val = available[i]
                            if val is not None and self._is_valid_assignment(r, c, val):
                                individual[r][c] = val
                                available[i] = None
                                if solve_block(idx + 1):
                                    return True
                                individual[r][c] = 0
                                available[i] = val
                        return False

                    if not solve_block(0):
                        available_left = [v for v in available if v is not None]
                        for r, c in non_fixed_coords:
                            if individual[r][c] == 0 and available_left:
                                individual[r][c] = available_left.pop()
            population.append(individual)
        return population

    def _compute_conflicts(self, grid):
        # Calculates raw conflict count (number of missing unique numbers in rows, cols, and blocks)
        conflicts = 0
        for i in range(GRID_SIZE):
            conflicts += (GRID_SIZE - len(set(grid[i, :])))
            conflicts += (GRID_SIZE - len(set(grid[:, i])))
        for i in range(3):
            for j in range(3):
                block = grid[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3].flatten()
                conflicts += (GRID_SIZE - len(set(block)))
        return conflicts

    def _compute_fitness(self, grid):
        # Objetivo teórico máximo: 243 (81 celdas únicas en filas, columnas y bloques)
        raw_conflicts = self._compute_conflicts(grid)
        if raw_conflicts == 0:
            return 243

        # Castigo severo por chocar contra pistas fijas (doradas).
        # Una pista dorada jamás se moverá; cualquier número de la IA que colisione
        # con una fija en su fila o columna representa una estructura inviable.
        fixed_conflicts = 0
        for r, c in self.non_fixed_positions:
            val = grid[r, c]
            if val in self.fixed_rows[r]:
                fixed_conflicts += 1
            if val in self.fixed_cols[c]:
                fixed_conflicts += 1

        # Cada colisión contra pista fija resta 3 puntos adicionales de fitness
        return 243 - raw_conflicts - (fixed_conflicts * 3)

    def _crossover_blocks(self, p1, p2):
        child = np.copy(p1)
        for i in range(3):
            for j in range(3):
                if np.random.rand() < 0.5:
                    for r in range(i * 3, (i + 1) * 3):
                        for c in range(j * 3, (j + 1) * 3):
                            if (r, c) not in self.fixed_positions:
                                child[r, c] = p2[r, c]
        return child

    def _crossover_uniform(self, p1, p2):
        """Crossover uniforme celda a celda para parejas con alta diversidad.
        Mezcla valores de ambos padres con probabilidad 50% por celda,
        permitiendo exploración más fina que el intercambio de bloques completos."""
        child = np.copy(p1)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) not in self.fixed_positions and np.random.rand() < 0.5:
                    child[r, c] = p2[r, c]
        return child

    def _mutate2(self, grid, max_attempts=5):
        mutated = np.copy(grid)
        scaled_attempts = max(1, int(max_attempts * (self.mutation_rate / 0.05)))

        # 1. Conflict Heatmap: identify blocks that actually have row/col collisions
        conflict_blocks = []
        for br in range(3):
            for bc in range(3):
                r_s, c_s = br * 3, bc * 3
                has_conf = False
                for r in range(r_s, r_s + 3):
                    for c in range(c_s, c_s + 3):
                        v = grid[r, c]
                        if np.count_nonzero(grid[r, :] == v) > 1 or np.count_nonzero(grid[:, c] == v) > 1:
                            has_conf = True
                            break
                    if has_conf:
                        break
                if has_conf:
                    conflict_blocks.append((br, bc))

        # Decide number of swaps based on mutation rate
        num_swaps = 1
        if self.mutation_rate > 0.08:
            num_swaps = np.random.randint(1, max(2, int(self.mutation_rate * 25)))
        
        swaps_done = 0
        attempts = 0
        while swaps_done < num_swaps and attempts < num_swaps * 10:
            attempts += 1
            if conflict_blocks and np.random.rand() < 0.85:
                # 85% probability: focus mutation directly on blocks with conflicts
                block_row, block_col = random.choice(conflict_blocks)
            else:
                block_row = np.random.randint(0, 3)
                block_col = np.random.randint(0, 3)

            r_start, c_start = block_row * 3, block_col * 3

            block_coords = [
                (r, c)
                for r in range(r_start, r_start + 3)
                for c in range(c_start, c_start + 3)
                if (r, c) not in self.fixed_positions
            ]

            if len(block_coords) < 2:
                continue

            # Pick k cells to shuffle (2, 3, or 4 cells)
            k = min(len(block_coords), random.randint(2, 4))
            
            # Prioritize cells with conflicts inside this block
            conf_cells = [
                (r, c) for r, c in block_coords
                if np.count_nonzero(mutated[r, :] == mutated[r, c]) > 1 or np.count_nonzero(mutated[:, c] == mutated[r, c]) > 1
            ]
            
            if conf_cells:
                # Include at least 1 conflict cell
                first_cell = random.choice(conf_cells)
                other_cells = random.sample([c for c in block_coords if c != first_cell], k - 1)
                cells = [first_cell] + other_cells
            else:
                cells = random.sample(block_coords, k)

            original_vals = [mutated[r, c] for r, c in cells]
            
            # Find a valid permutation (try up to 5 times)
            valid_found = False
            new_vals = list(original_vals)
            for _ in range(5):
                random.shuffle(new_vals)
                if new_vals == original_vals:
                    continue
                # Hard fixed-clue constraint filter
                valid = True
                for i, (r, c) in enumerate(cells):
                    if not self._is_valid_assignment(r, c, new_vals[i]):
                        valid = False
                        break
                if valid:
                    valid_found = True
                    break
                    
            if not valid_found:
                continue
                
            old_fitness = self._compute_fitness(mutated)
            for i, (r, c) in enumerate(cells):
                mutated[r, c] = new_vals[i]
            new_fitness = self._compute_fitness(mutated)

            # Tabu check: if new state was visited recently, revert unless fitness improves
            if self._is_tabu(mutated) and new_fitness <= old_fitness:
                for i, (r, c) in enumerate(cells):
                    mutated[r, c] = original_vals[i]
                continue

            # Accept if better/equal, or by random chance based on mutation rate
            if new_fitness >= old_fitness or np.random.rand() < max(0.20, self.mutation_rate):
                self._add_tabu(mutated)
                swaps_done += 1
            else:
                for i, (r, c) in enumerate(cells):
                    mutated[r, c] = original_vals[i]

        return mutated

    def _block_has_val(self, grid, r, c, val):
        """Devuelve True si 'val' ya existe en el bloque 3x3 que contiene (r, c),
        ignorando la celda (r, c) misma. Necesario para garantizar unicidad al
        hacer swaps inter-bloque sin violar la restricción de subcuadrícula."""
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                rr, cc = br + dr, bc + dc
                if (rr, cc) != (r, c) and grid[rr, cc] == val:
                    return True
        return False

    def _mutate_cross_block(self, grid):
        """Operador de mutación inter-bloque: intercambia celdas entre distintos
        bloques 3x3 que comparten la misma fila o columna. Es el único operador
        capaz de resolver los últimos 1-2 conflictos de fila/columna que quedan
        cuando los bloques ya son perfectos internamente."""
        mutated = np.copy(grid)

        # Determinar si trabajar sobre fila o columna conflictiva
        bad_rows = [r for r in range(GRID_SIZE) if len(set(mutated[r, :])) < GRID_SIZE]
        bad_cols = [c for c in range(GRID_SIZE) if len(set(mutated[:, c])) < GRID_SIZE]

        # Elegir modo: fila o columna (preferir donde haya conflicto)
        if bad_rows and (not bad_cols or np.random.rand() < 0.5):
            row = random.choice(bad_rows)
            # Candidatos: celdas no fijas en esa fila
            candidates = [(row, c) for c in range(GRID_SIZE)
                          if (row, c) not in self.fixed_positions]
            if len(candidates) < 2:
                return mutated
            # Elegir dos celdas de bloques 3x3 DISTINTOS para el swap
            for _ in range(20):
                (r1, c1), (r2, c2) = random.sample(candidates, 2)
                # Asegurar que son bloques distintos (c1//3 != c2//3)
                if c1 // 3 == c2 // 3:
                    continue
                v1, v2 = mutated[r1, c1], mutated[r2, c2]
                if v1 == v2:
                    continue
                # Verificar: no crear conflicto de bloque en el destino
                if (self._is_valid_assignment(r1, c1, v2) and
                        self._is_valid_assignment(r2, c2, v1) and
                        not self._block_has_val(mutated, r1, c1, v2) and
                        not self._block_has_val(mutated, r2, c2, v1)):
                    mutated[r1, c1], mutated[r2, c2] = v2, v1
                    break
        elif bad_cols:
            col = random.choice(bad_cols)
            candidates = [(r, col) for r in range(GRID_SIZE)
                          if (r, col) not in self.fixed_positions]
            if len(candidates) < 2:
                return mutated
            for _ in range(20):
                (r1, c1), (r2, c2) = random.sample(candidates, 2)
                # Asegurar que son bloques distintos (r1//3 != r2//3)
                if r1 // 3 == r2 // 3:
                    continue
                v1, v2 = mutated[r1, c1], mutated[r2, c2]
                if v1 == v2:
                    continue
                # Verificar: no crear conflicto de bloque en el destino
                if (self._is_valid_assignment(r1, c1, v2) and
                        self._is_valid_assignment(r2, c2, v1) and
                        not self._block_has_val(mutated, r1, c1, v2) and
                        not self._block_has_val(mutated, r2, c2, v1)):
                    mutated[r1, c1], mutated[r2, c2] = v2, v1
                    break
        return mutated

    def _get_diverse_top9(self, population, fitness_scores):
        if not population:
            return []

        # Leader #1 is always the absolute best individual
        selected_indices = [0]
        selected_grids = [population[0]]

        # Greedily pick up to 8 contenders with genuine Hamming distance
        for min_dist in [8, 6, 4, 2]:
            if len(selected_indices) >= 9:
                break
            for idx in range(1, len(population)):
                if len(selected_indices) >= 9:
                    break
                if idx in selected_indices:
                    continue
                cand = population[idx]
                if all(np.count_nonzero(cand != prev) >= min_dist for prev in selected_grids):
                    selected_indices.append(idx)
                    selected_grids.append(cand)

        # Fill any remaining slots
        for idx in range(1, len(population)):
            if len(selected_indices) >= 9:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)
                selected_grids.append(population[idx])

        top9_payload = []
        for idx in selected_indices:
            ind = population[idx]
            fit = fitness_scores[idx] if idx < len(fitness_scores) else self._compute_fitness(ind)
            top9_payload.append({
                "grid": ind.flatten().tolist(),
                "fitness": int(fit),
                "conflicts": int(self._compute_conflicts(ind))
            })
        return top9_payload

    def _local_improvement(self, individual, max_passes=8):
        improved = np.copy(individual)
        # Multi-pass Min-Conflicts with Sideway Moves (Plateau traversal)
        for _ in range(max_passes):
            if self._compute_conflicts(improved) == 0:
                break

            pass_modified = False

            for block_row in range(3):
                for block_col in range(3):
                    r_start, c_start = block_row * 3, block_col * 3
                    non_fixed = [
                        (r, c)
                        for r in range(r_start, r_start + 3)
                        for c in range(c_start, c_start + 3)
                        if (r, c) not in self.fixed_positions
                    ]
                    if len(non_fixed) < 2:
                        continue

                    best_strict_swap = None
                    best_gain = 0
                    neutral_swaps = []

                    for i in range(len(non_fixed)):
                        r1, c1 = non_fixed[i]
                        val1 = improved[r1, c1]
                        c1_conf = (np.count_nonzero(improved[r1, :] == val1) > 1) or \
                                  (np.count_nonzero(improved[:, c1] == val1) > 1)

                        for j in range(i + 1, len(non_fixed)):
                            r2, c2 = non_fixed[j]
                            val2 = improved[r2, c2]
                            if val1 == val2:
                                continue

                            c2_conf = (np.count_nonzero(improved[r2, :] == val2) > 1) or \
                                      (np.count_nonzero(improved[:, c2] == val2) > 1)

                            # At least one cell must be in conflict
                            if not (c1_conf or c2_conf):
                                continue

                            # Hard fixed-clue constraint filter:
                            if not (self._is_valid_assignment(r1, c1, val2) and self._is_valid_assignment(r2, c2, val1)):
                                continue

                            before = (
                                (9 - len(set(improved[r1, :]))) +
                                (9 - len(set(improved[:, c1]))) +
                                (9 - len(set(improved[r2, :]))) +
                                (9 - len(set(improved[:, c2])))
                            )

                            improved[r1, c1], improved[r2, c2] = val2, val1
                            after = (
                                (9 - len(set(improved[r1, :]))) +
                                (9 - len(set(improved[:, c1]))) +
                                (9 - len(set(improved[r2, :]))) +
                                (9 - len(set(improved[:, c2])))
                            )
                            improved[r1, c1], improved[r2, c2] = val1, val2

                            gain = before - after
                            if gain > best_gain:
                                best_gain = gain
                                best_strict_swap = (r1, c1, r2, c2)
                            elif gain == 0 and best_gain == 0:
                                neutral_swaps.append((r1, c1, r2, c2))

                    if best_strict_swap is not None and best_gain > 0:
                        r1, c1, r2, c2 = best_strict_swap
                        improved[r1, c1], improved[r2, c2] = improved[r2, c2], improved[r1, c1]
                        pass_modified = True
                    elif neutral_swaps and np.random.rand() < 0.85:
                        # Sideway plateau step: 85% probability to move laterally across 0-gain plateaus
                        r1, c1, r2, c2 = random.choice(neutral_swaps)
                        improved[r1, c1], improved[r2, c2] = improved[r2, c2], improved[r1, c1]
                        pass_modified = True

            if not pass_modified:
                break

        return improved

    def _propagate_from_blocks(self, grid, chosen_blocks):
        """Propagación lógica por efecto dominó a partir de bloques ancla:
        Fija los bloques seleccionados junto con las pistas originales del puzzle,
        vacía los bloques restantes y aplica deducción de candidatos únicos (Naked Singles)
        y búsqueda MRV rápida.
        
        Devuelve el tablero resuelto (0 conflictos) si la combinación es consistente,
        o None si se produce una contradicción lógica."""
        test_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        # 1. Copiar las pistas fijas originales en todo el tablero
        for (r, c) in self.fixed_positions:
            test_grid[r, c] = int(self.sudoku_puzzle[r, c])

        # 2. Copiar todas las celdas de los bloques elegidos
        for (br, bc) in chosen_blocks:
            for r in range(br * 3, (br + 1) * 3):
                for c in range(bc * 3, (bc + 1) * 3):
                    test_grid[r, c] = int(grid[r, c])

        # Verificar que no haya colisiones inmediatas entre los bloques elegidos
        for i in range(GRID_SIZE):
            row_vals = [v for v in test_grid[i, :] if v != 0]
            if len(row_vals) != len(set(row_vals)):
                return None
            col_vals = [v for v in test_grid[:, i] if v != 0]
            if len(col_vals) != len(set(col_vals)):
                return None

        # 3. Propagación lógica por Naked Singles (Efecto Dominó)
        def get_candidates(g, r, c):
            used = set(g[r, :]) | set(g[:, c])
            br, bc = (r // 3) * 3, (c // 3) * 3
            used |= set(g[br:br + 3, bc:bc + 3].flatten())
            return set(range(1, 10)) - used

        changed = True
        while changed:
            changed = False
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if test_grid[r, c] == 0:
                        cands = get_candidates(test_grid, r, c)
                        if len(cands) == 0:
                            return None  # Contradicción: rama muerta
                        if len(cands) == 1:
                            test_grid[r, c] = cands.pop()
                            changed = True

        empty_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if test_grid[r, c] == 0]
        if not empty_cells:
            return test_grid  # ¡Resuelto al 100% solo con dominó lógico!

        if len(empty_cells) > 55:
            return None  # No hay suficientes pistas para acotar la búsqueda

        nodes = [0]
        max_nodes = 5000

        def solve_mrv():
            nodes[0] += 1
            if nodes[0] > max_nodes:
                return False

            best_cell = None
            best_cands = None
            min_c = 10

            for (r, c) in empty_cells:
                if test_grid[r, c] == 0:
                    cands = list(get_candidates(test_grid, r, c))
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_c:
                        min_c = len(cands)
                        best_cell = (r, c)
                        best_cands = cands
                        if min_c == 1:
                            break

            if best_cell is None:
                return True

            r, c = best_cell
            for val in best_cands:
                test_grid[r, c] = val
                if solve_mrv():
                    return True
                test_grid[r, c] = 0

            return False

        if solve_mrv():
            return test_grid
        return None

    def _test_and_solve_viability(self, grid):
        """Test de viabilidad por CSP con heurística MRV:
        Vacia las casillas involucradas en los conflictos de filas y columnas,
        e intenta completarlas respetando todas las reglas del Sudoku.
        
        - Si la estructura verde es consistente, devuelve la solución perfecta (0 conflictos).
        - Si la estructura verde es contradictoria, devuelve None."""
        bad_rows = {r for r in range(GRID_SIZE) if len(set(grid[r, :])) < GRID_SIZE}
        bad_cols = {c for c in range(GRID_SIZE) if len(set(grid[:, c])) < GRID_SIZE}

        suspect_cells = [
            (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
            if (r, c) not in self.fixed_positions and (r in bad_rows or c in bad_cols)
        ]

        if not suspect_cells:
            return grid if self._compute_conflicts(grid) == 0 else None

        if len(suspect_cells) > 35:
            return None

        test_grid = np.copy(grid)
        for r, c in suspect_cells:
            test_grid[r, c] = 0

        def is_legal(g, r, c, val):
            if val in g[r, :]:
                return False
            if val in g[:, c]:
                return False
            br, bc = (r // 3) * 3, (c // 3) * 3
            if val in g[br:br + 3, bc:bc + 3]:
                return False
            return True

        nodes = [0]
        max_nodes = 3000

        def solve_mrv():
            nodes[0] += 1
            if nodes[0] > max_nodes:
                return False

            best_cell = None
            best_candidates = None
            min_cands = 10

            for (r, c) in suspect_cells:
                if test_grid[r, c] == 0:
                    cands = [v for v in range(1, 10) if is_legal(test_grid, r, c, v)]
                    if len(cands) == 0:
                        return False
                    if len(cands) < min_cands:
                        min_cands = len(cands)
                        best_cell = (r, c)
                        best_candidates = cands
                        if min_cands == 1:
                            break

            if best_cell is None:
                return True

            r, c = best_cell
            for val in best_candidates:
                test_grid[r, c] = val
                if solve_mrv():
                    return True
                test_grid[r, c] = 0

            return False

        if solve_mrv():
            return test_grid
        return None

    def _endgame_resolver(self, grid):
        """Búsqueda exhaustiva multi-fase para los últimos 2-6 conflictos:
        1. Propagación lógica por bloques ancla (efecto dominó).
        2. Test de viabilidad por CSP con MRV.
        3. Swaps locales de contingencia."""
        conflicts = self._compute_conflicts(grid)
        if conflicts == 0:
            return grid

        # 1. Probar propagación de dominó lógico en los 9 tríos geométricos de bloques
        trios = [
            self.anchor_blocks,
            [(0, 0), (1, 1), (2, 2)],  # Diagonal principal
            [(0, 2), (1, 1), (2, 0)],  # Anti-diagonal
            [(0, 0), (0, 1), (0, 2)],  # Banda superior
            [(1, 0), (1, 1), (1, 2)],  # Banda central
            [(2, 0), (2, 1), (2, 2)],  # Banda inferior
            [(0, 0), (1, 0), (2, 0)],  # Pila izquierda
            [(0, 1), (1, 1), (2, 1)],  # Pila central
            [(0, 2), (1, 2), (2, 2)],  # Pila derecha
        ]
        for trio in trios:
            solved = self._propagate_from_blocks(grid, trio)
            if solved is not None and self._compute_conflicts(solved) == 0:
                return solved

        # 2. Test de viabilidad por CSP sobre casillas conflictivas si conflicts <= 6
        if conflicts <= 6:
            solved = self._test_and_solve_viability(grid)
            if solved is not None and self._compute_conflicts(solved) == 0:
                return solved

        improved = np.copy(grid)
        if conflicts > 6:
            return improved

        def get_conflict_cells(g):
            cells = []
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) in self.fixed_positions:
                        continue
                    v = g[r, c]
                    if (np.count_nonzero(g[r, :] == v) > 1 or
                            np.count_nonzero(g[:, c] == v) > 1):
                        cells.append((r, c))
            return cells

        def try_swap(g, r1, c1, r2, c2):
            """Intenta un swap y devuelve (new_conflicts, revert_fn)."""
            v1, v2 = g[r1, c1], g[r2, c2]
            if v1 == v2:
                return None
            if not (self._is_valid_assignment(r1, c1, v2) and
                    self._is_valid_assignment(r2, c2, v1)):
                return None
            g[r1, c1], g[r2, c2] = v2, v1
            nc = self._compute_conflicts(g)
            g[r1, c1], g[r2, c2] = v1, v2
            return nc

        changed = True
        while changed and conflicts > 0:
            changed = False
            conflict_cells = get_conflict_cells(improved)
            if not conflict_cells:
                break

            # ── Phase 1: swaps entre celdas en conflicto ─────────────────────
            best_conf, best_swap = conflicts, None
            for i in range(len(conflict_cells)):
                r1, c1 = conflict_cells[i]
                for j in range(i + 1, len(conflict_cells)):
                    r2, c2 = conflict_cells[j]
                    nc = try_swap(improved, r1, c1, r2, c2)
                    if nc is not None and nc < best_conf:
                        best_conf, best_swap = nc, (r1, c1, r2, c2)

            if best_swap:
                r1, c1, r2, c2 = best_swap
                improved[r1, c1], improved[r2, c2] = improved[r2, c2], improved[r1, c1]
                conflicts = best_conf
                changed = True
                if conflicts == 0:
                    return improved
                continue

            # ── Phase 2: swap conflicto vs no-conflicto ───────────────────────
            all_free = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
                        if (r, c) not in self.fixed_positions and (r, c) not in conflict_cells]
            best_conf2, best_swap2 = conflicts, None
            for (r1, c1) in conflict_cells:
                for (r2, c2) in all_free:
                    nc = try_swap(improved, r1, c1, r2, c2)
                    if nc is not None and nc < best_conf2:
                        best_conf2, best_swap2 = nc, (r1, c1, r2, c2)

            if best_swap2:
                r1, c1, r2, c2 = best_swap2
                improved[r1, c1], improved[r2, c2] = improved[r2, c2], improved[r1, c1]
                conflicts = best_conf2
                changed = True
                if conflicts == 0:
                    return improved
                continue



        return improved

    def _ruin_and_recreate(self, grid, num_blocks=None):
        """Reorganización intra-bloque de celdas en conflicto:
        En vez de borrar a 0 y regenerar por backtracking, busca las celdas
        en conflicto (rojas) y las intercambia (swap) con otras celdas no fijas
        dentro de su mismo bloque 3x3.
        
        Ventajas:
        1. Mantiene el bloque 3x3 100% válido (conserva todos los números 1..9).
        2. Mueve los números conflictivos a nuevas filas y columnas.
        3. Desplaza celdas no conflictivas (verdes) que podían ser falsos positivos."""
        ruined = np.copy(grid)

        # Recorremos cada uno de los 9 bloques 3x3
        for block_row in range(3):
            for block_col in range(3):
                r_start, c_start = block_row * 3, block_col * 3
                block_coords = [
                    (r, c)
                    for r in range(r_start, r_start + 3)
                    for c in range(c_start, c_start + 3)
                    if (r, c) not in self.fixed_positions
                ]
                if len(block_coords) < 2:
                    continue

                # Identificar qué celdas no fijas de este bloque están en conflicto de fila o columna
                conf_cells = [
                    (r, c) for (r, c) in block_coords
                    if (np.count_nonzero(ruined[r, :] == ruined[r, c]) > 1 or
                        np.count_nonzero(ruined[:, c] == ruined[r, c]) > 1)
                ]

                if not conf_cells:
                    continue

                # Barajamos las celdas en conflicto para que cada clon tome órdenes distintos
                random.shuffle(conf_cells)

                for (r1, c1) in conf_cells:
                    # Se intercambia incondicionalmente con cualquier celda verde (sin conflicto previo) del bloque
                    green_cells = [c for c in block_coords if c not in conf_cells]
                    if green_cells:
                        r2, c2 = random.choice(green_cells)
                    else:
                        other_cells = [c for c in block_coords if c != (r1, c1)]
                        if not other_cells:
                            continue
                        r2, c2 = random.choice(other_cells)

                    # Intercambio incondicional dentro del bloque 3x3
                    ruined[r1, c1], ruined[r2, c2] = ruined[r2, c2], ruined[r1, c1]

        # Paso 2: Barajado forzado de 1 o 2 bloques completos (rompe jaulas verdes como el bloque del centro)
        all_blocks = [(br, bc) for br in range(3) for bc in range(3)]
        chosen_blocks = random.sample(all_blocks, k=random.randint(1, 2))
        for (br, bc) in chosen_blocks:
            r_start, c_start = br * 3, bc * 3
            non_fixed_block = [
                (r, c)
                for r in range(r_start, r_start + 3)
                for c in range(c_start, c_start + 3)
                if (r, c) not in self.fixed_positions
            ]
            if len(non_fixed_block) >= 2:
                vals = [ruined[r, c] for r, c in non_fixed_block]
                np.random.shuffle(vals)
                for i, (r, c) in enumerate(non_fixed_block):
                    ruined[r, c] = vals[i]

        return ruined

    def create_individual(self, grid):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(grid)))

    def remove_duplicates(self, population):
        seen = set()
        unique_population = []
        for individual in population:
            hash_id = self.create_individual(individual)
            if hash_id not in seen:
                seen.add(hash_id)
                unique_population.append(individual)
        return unique_population

    def handle_command(self, cmd_obj):
        cmd = cmd_obj.get("cmd")
        if cmd == "pause":
            self.paused = True
        elif cmd == "resume":
            self.paused = False
        elif cmd == "toggle_pause":
            self.paused = not self.paused
        elif cmd == "step":
            self.paused = True
            self.step_mode = True
        elif cmd == "speed":
            self.delay = max(0.0, float(cmd_obj.get("val", 0.02)))
        elif cmd == "mutate_burst":
            self.mutation_rate = min(0.35, self.mutation_rate + 0.08)
        elif cmd == "reset":
            return "RESET"
        return None

    def start_genetic_algorithm(self):
        # Emit initial puzzle layout and fixed cell mask
        fixed_mask = [1 if (r, c) in self.fixed_positions else 0 for r in range(9) for c in range(9)]
        self.emit({
            "type": "init",
            "fixed": fixed_mask,
            "puzzle": self.sudoku_puzzle.flatten().tolist()
        })

        population = self._generate_population_blocks(self.pop_size)
        no_improve_count = 0
        best_fitness = 0
        self.history_best = set()

        import time

        for generation in range(1, self.max_generations + 1):
            # Check for incoming commands on stdin
            while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    break
                try:
                    cmd_obj = json.loads(line.strip())
                    res = self.handle_command(cmd_obj)
                    if res == "RESET":
                        return "RESET"
                except Exception:
                    pass

            while self.paused and not self.step_mode:
                time.sleep(0.05)
                while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    try:
                        cmd_obj = json.loads(line.strip())
                        res = self.handle_command(cmd_obj)
                        if res == "RESET":
                            return "RESET"
                    except Exception:
                        pass

            self.step_mode = False

            # Sort population descending by fitness
            fitness_tuples = [(ind, self._compute_fitness(ind)) for ind in population]
            fitness_tuples.sort(key=lambda x: -x[1])
            population = [item[0] for item in fitness_tuples]
            fitness_scores = [item[1] for item in fitness_tuples]

            # Memetic local improvement + endgame resolver sobre el top 3 de la población
            for idx in range(min(3, len(population))):
                cand = self._local_improvement(population[idx])
                cand = self._endgame_resolver(cand)
                population[idx] = cand
                if self._compute_conflicts(cand) == 0:
                    population[0] = cand
                    break

            best_solution = population[0]
            raw_conf = self._compute_conflicts(best_solution)
            is_solved = (raw_conf == 0)
            if is_solved:
                current_fitness = 243
            else:
                current_fitness = self._compute_fitness(best_solution)

            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.history_best.add(self.create_individual(best_solution))

            if current_fitness <= best_fitness:
                no_improve_count += 1
            else:
                best_fitness = current_fitness
                no_improve_count = 0
                # Gently decay mutation rate when finding improvements
                self.mutation_rate = max(self.initial_mutation_rate, self.mutation_rate * 0.92)

            # Fast-Decay: umbrales cortos para no malgastar ciclos en linajes muertos
            if current_fitness >= 241:
                stagnation_threshold = 25   # Linaje en 241: 25 generaciones es suficiente
            elif current_fitness >= 238:
                stagnation_threshold = 20
            else:
                stagnation_threshold = 15

            stagnation_triggered = False
            if no_improve_count >= stagnation_threshold:
                stagnation_triggered = True
                no_improve_count = 0

            # Build Top 9 payload with Hamming distance diversity
            top9_payload = self._get_diverse_top9(population, fitness_scores)

            # Emit generation packet to Rust
            self.emit({
                "type": "generation",
                "gen": generation,
                "best_fitness": int(current_fitness),
                "conflicts": int(raw_conf),
                "avg_fitness": round(float(avg_fitness), 2),
                "mutation_rate": round(float(self.mutation_rate), 4),
                "stagnation": no_improve_count,
                "stagnation_boost": stagnation_triggered,
                "solved": is_solved,
                "top9": top9_payload
            })

            if is_solved:
                self.emit({"type": "solved", "gen": generation})
                # Keep alive until user resets or quits
                while True:
                    time.sleep(0.1)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        if line:
                            try:
                                cmd = json.loads(line.strip())
                                if cmd.get("cmd") == "reset":
                                    return "RESET"
                            except Exception:
                                pass

            # EXTINCIÓN TOTAL: Si el linaje se estancó, se erradica al 100% de la población (incluyendo al líder)
            if stagnation_triggered:
                # Reset completo: nueva población desde cero, sin líderes zombis ni veneno de tabú
                population = self._generate_population_blocks(self.pop_size)
                self.confidence_matrix.fill(0.0)
                best_fitness = 0
                no_improve_count = 0
                continue

            # Selection (top 50%)
            parents = population[:max(2, int(self.pop_size * 0.50))]

            # Elite preservation (5% normal)
            elite_count = max(1, int(0.05 * self.pop_size))
            elite = []
            for e in population[:elite_count]:
                elite.append(self._local_improvement(e))

            # 1. Leader + Elites
            new_pop = [best_solution] + elite

            # 2. Inmigración balanceada (15%) para mantener diversidad genética
            immigrant_count = int(self.pop_size * 0.15)
            immigrants = self._generate_population_blocks(immigrant_count)
            new_pop.extend(immigrants)

            # 3. Crossover + Mutation
            while len(new_pop) < self.pop_size:
                p1, p2 = random.choices(parents, k=2)
                hamming = int(np.count_nonzero(p1 != p2))
                use_uniform = (hamming > 20) or (current_fitness >= 238 and np.random.rand() < 0.40)
                if use_uniform:
                    crossed = self._crossover_uniform(p1, p2)
                else:
                    crossed = self._crossover_blocks(p1, p2)

                cross_block_prob = 0.50 if current_fitness >= 240 else 0.25
                if np.random.rand() < cross_block_prob:
                    child = self._mutate_cross_block(self._mutate2(crossed))
                else:
                    child = self._mutate2(crossed)
                new_pop.append(child)

            population = self.remove_duplicates(new_pop)

            # Refill population if duplicates were pruned to prevent population collapse
            if len(population) < self.pop_size:
                population.extend(self._generate_population_blocks(self.pop_size - len(population)))

            if self.delay > 0:
                time.sleep(self.delay)

        return "MAX_GEN_REACHED"


def main():
    import time
    emit_puzzles_catalog()

    current_puzzle_key = "easy"
    initial_delay = 0.40

    while True:
        # Idle state: waiting for user to click START in GUI
        sys.stdout.write(json.dumps({"type": "idle", "selected": current_puzzle_key}) + "\n")
        sys.stdout.flush()

        started = False
        while not started:
            time.sleep(0.03)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    return
                try:
                    cmd_obj = json.loads(line.strip())
                    cmd = cmd_obj.get("cmd")
                    if cmd == "start":
                        current_puzzle_key = cmd_obj.get("puzzle", current_puzzle_key)
                        if "delay" in cmd_obj:
                            initial_delay = float(cmd_obj["delay"])
                        started = True
                    elif cmd == "select_puzzle":
                        current_puzzle_key = cmd_obj.get("puzzle", current_puzzle_key)
                        sys.stdout.write(json.dumps({"type": "puzzle_selected", "puzzle": current_puzzle_key}) + "\n")
                        sys.stdout.flush()
                except Exception:
                    pass

        selected_puzzle = PUZZLES.get(current_puzzle_key, PUZZLES["easy"])
        ga = SudokuGeneticAlgorithm(
            pop_size=120,
            max_generations=5000,
            mutation_rate=0.015,
            sudoku_puzzle=selected_puzzle
        )
        ga.delay = initial_delay
        ga.start_genetic_algorithm()


if __name__ == "__main__":
    main()
