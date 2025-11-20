import numpy as np
import adios2
import os
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances


# ============================================================
# Funções de I/O com ADIOS2
# ============================================================

def write_fragment_bp(data, fname):
    with adios2.open(fname, "w") as fh:
        fh.write("fragment", data)


def read_fragment_bp(fname):
    with adios2.open(fname, "r") as fh:
        for step in fh:
            return step.read("fragment")


def write_partial_bp(partial, fname):
    with adios2.open(fname, "w") as fh:
        fh.write("partial", partial)


def read_partial_bp(fname):
    with adios2.open(fname, "r") as fh:
        for step in fh:
            return step.read("partial")


# ============================================================
# Partial Sum (equivalente ao @task PyCOMPSs)
# ============================================================

def partial_sum_bp(frag_filename, centres, outname):
    fragment = read_fragment_bp(frag_filename)

    partials = np.zeros((centres.shape[0], 2), dtype=object)

    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)

    for center_idx in range(centres.shape[0]):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]

    write_partial_bp(partials, outname)
    return outname


# ============================================================
# Merge (redução)
# ============================================================

def merge_bp(input_files, output_file):
    accum = read_partial_bp(input_files[0]).copy()

    for fname in input_files[1:]:
        d = read_partial_bp(fname)
        accum += d

    write_partial_bp(accum, output_file)
    return output_file


# ============================================================
# Recompute centres a partir dos parciais
# ============================================================

def recompute_centres_bp(partial_files, old_centres, arity):
    centres = old_centres.copy()

    # Redução em árvore usando ADIOS2 (como PyCOMPSs faz)
    files = list(partial_files)

    while len(files) > 1:
        subset = files[:arity]
        files = files[arity:]
        merged_name = f"merged_{len(files)}.bp"
        files.append(merge_bp(subset, merged_name))

    final_partial = read_partial_bp(files[0])

    # Recalcular centros
    for idx, sum_ in enumerate(final_partial):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]

    return centres


# ============================================================
# Critério de convergência
# ============================================================

def converged(old_centres, centres, epsilon, iteration, max_iter):
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


# ============================================================
# Algoritmo K-Means usando ADIOS2
# ============================================================

def kmeans_frag_adios2(fragment_files, dimensions, num_centres=10,
                       iterations=20, seed=0, epsilon=1e-9, arity=50):

    np.random.seed(seed)
    centres = np.random.random((num_centres, dimensions))
    old_centres = None
    iteration = 0

    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print(f"Iteração {iteration+1}/{iterations}")
        old_centres = centres.copy()

        partial_files = []
        for i, frag in enumerate(fragment_files):
            partial_name = f"partial_{iteration}_{i}.bp"
            partial_files.append(
                partial_sum_bp(frag, old_centres, partial_name)
            )

        centres = recompute_centres_bp(partial_files, old_centres, arity)
        iteration += 1

    return centres


# ============================================================
# Fragment generator (equivalente ao @task PyCOMPSs)
# ============================================================

def generate_fragment_bp(points, dim, mode, seed, outname):
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)

    mat = np.asarray([r(dim) for _ in range(points)])

    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx

    write_fragment_bp(mat, outname)
    return outname


# ============================================================
# Main
# ============================================================

def run_adios2_kmeans(seed=0, numpoints=100, dimensions=2,
                      num_centres=5, fragments=10, mode="uniform",
                      iterations=20, epsilon=1e-9, arity=50):

    print("Gerando fragmentos...")

    fragment_files = []
    points_per_fragment = max(1, numpoints // fragments)

    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fname = f"fragment_{l}.bp"
        generate_fragment_bp(r - l, dimensions, mode, seed + l, fname)
        fragment_files.append(fname)

    print("Iniciando K-Means com ADIOS2...")

    centres = kmeans_frag_adios2(
        fragment_files,
        dimensions,
        num_centres,
        iterations,
        seed,
        epsilon,
        arity,
    )

    print("Centros finais:")
    print(centres)

    return centres


if __name__ == "__main__":
    run_adios2_kmeans()
