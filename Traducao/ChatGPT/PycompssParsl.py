import numpy as np
import time
import parsl
from parsl import python_app, bash_app
from parsl.configs.local_threads import config as local_config

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

# Inicializa Parsl
parsl.load(local_config)


# ---------------------------------------------------------
#  APPS PARALLELAS
# ---------------------------------------------------------

@python_app
def partial_sum(fragment, centres):
    import numpy as np
    from sklearn.metrics import pairwise_distances

    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    return partials


@python_app
def merge(*data):
    import numpy as np
    accum = data[0].copy()
    for d in data[1:]:
        accum += d
    return accum


@python_app
def generate_fragment(points, dim, mode, seed):
    import numpy as np
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }

    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for _ in range(points)])
    # Normalização
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    return mat


# ---------------------------------------------------------
#  FUNÇÕES AUXILIARES (MASTER SIDE)
# ---------------------------------------------------------

def converged(old_centres, centres, epsilon, iteration, max_iter):
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


def recompute_centres(partials_futures, old_centres, arity):
    centres = old_centres.copy()

    # Redução manual de futures
    futures = partials_futures
    while len(futures) > 1:
        subset = futures[:arity]
        futures = futures[arity:]
        futures.append(merge(*subset))

    # Espera o único futuro final
    final_partials = futures[0].result()

    for idx, sum_ in enumerate(final_partials):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
    return centres


# ---------------------------------------------------------
#  KMEANS FRAGMENTADO
# ---------------------------------------------------------

def kmeans_frag(
    fragments,
    dimensions,
    num_centres=10,
    iterations=20,
    seed=0.0,
    epsilon=1e-9,
    arity=50,
):

    np.random.seed(seed)
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None

    iteration = 0

    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print(f"Doing iteration #{iteration+1}/{iterations}")
        old_centres = centres.copy()

        partials = []
        # partial_sum retorna futures
        for frag_future in fragments:
            partials.append(partial_sum(frag_future.result(), old_centres))

        centres = recompute_centres(partials, old_centres, arity)

        iteration += 1

    return centres


# ---------------------------------------------------------
#  MAIN + PARSER
# ---------------------------------------------------------

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="KMeans Clustering.")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--numpoints", type=int, default=100)
    parser.add_argument("-d", "--dimensions", type=int, default=2)
    parser.add_argument("-c", "--num_centres", type=int, default=5)
    parser.add_argument("-f", "--fragments", type=int, default=10)
    parser.add_argument("-m", "--mode", type=str, default="uniform",
                        choices=["uniform", "normal"])
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9)
    parser.add_argument("-a", "--arity", type=int, default=50)
    return parser.parse_args()


def main(
    seed,
    numpoints,
    dimensions,
    num_centres,
    fragments,
    mode,
    iterations,
    epsilon,
    arity,
):
    start_time = time.time()

    # Geração dos fragments em paralelo
    fragment_futures = []

    points_per_fragment = max(1, numpoints // fragments)
    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        fragment_futures.append(
            generate_fragment(r - l, dimensions, mode, seed + l)
        )

    print("Generation/Load done")
    initialization_time = time.time()

    print("Starting kmeans")
    centres = kmeans_frag(
        fragments=fragment_futures,
        dimensions=dimensions,
        num_centres=num_centres,
        iterations=iterations,
        seed=seed,
        epsilon=epsilon,
        arity=arity,
    )

    print("Ending kmeans")
    kmeans_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time:", initialization_time - start_time)
    print("KMeans time:", kmeans_time - initialization_time)
    print("Total time:", kmeans_time - start_time)
    print("-----------------------------------------")
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")


if __name__ == "__main__":
    args = parse_arguments()
    main(**vars(args))
