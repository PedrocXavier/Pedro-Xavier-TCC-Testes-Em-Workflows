import numpy as np
import time
import adios2
from mpi4py import MPI

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances


def partial_sum(fragment, centres):
    """
    Calcula somas parciais para cada centro
    """
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    return partials


def converged(old_centres, centres, epsilon, iteration, max_iter):
    """
    Verifica convergência do algoritmo
    """
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


def recompute_centres(all_partials, old_centres):
    """
    Recomputa centros a partir das somas parciais
    """
    centres = old_centres.copy()
    
    # Agregar todas as parciais
    accumulated = all_partials[0].copy()
    for partial in all_partials[1:]:
        accumulated += partial
    
    # Calcular novos centros
    for idx, sum_ in enumerate(accumulated):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
    
    return centres


def write_fragment_adios(writer, fragment, fragment_id, step):
    """
    Escreve um fragmento usando ADIOS2
    """
    var_name = f"fragment_{fragment_id}"
    shape = fragment.shape
    writer.write(var_name, fragment, shape, [0, 0], shape)


def read_fragment_adios(reader, fragment_id, step):
    """
    Lê um fragmento usando ADIOS2
    """
    var_name = f"fragment_{fragment_id}"
    fragment = reader.read(var_name)
    return fragment


def kmeans_adios(
    num_fragments,
    dimensions,
    numpoints,
    num_centres=10,
    iterations=20,
    seed=0.0,
    epsilon=1e-9,
    mode="uniform",
):
    """
    K-Means usando ADIOS2 para I/O paralelo
    
    :param num_fragments: Número de fragmentos
    :param dimensions: Número de dimensões
    :param numpoints: Número total de pontos
    :param num_centres: Número de centros
    :param iterations: Máximo de iterações
    :param seed: Semente aleatória
    :param epsilon: Epsilon (distância de convergência)
    :param mode: Modo de geração de dados
    :return: Centros finais
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Inicializar ADIOS2
    adios = adios2.ADIOS(comm)
    
    # Configurar centros iniciais (apenas no rank 0)
    if rank == 0:
        np.random.seed(seed)
        centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    else:
        centres = np.zeros((num_centres, dimensions))
    
    # Broadcast centros iniciais
    centres = comm.bcast(centres, root=0)
    
    old_centres = None
    iteration = 0
    
    # Determinar quais fragmentos este rank processa
    fragments_per_rank = num_fragments // size
    remainder = num_fragments % size
    
    if rank < remainder:
        start_frag = rank * (fragments_per_rank + 1)
        end_frag = start_frag + fragments_per_rank + 1
    else:
        start_frag = rank * fragments_per_rank + remainder
        end_frag = start_frag + fragments_per_rank
    
    local_fragments = list(range(start_frag, end_frag))
    
    # Gerar dados locais
    points_per_fragment = max(1, numpoints // num_fragments)
    fragments_data = []
    
    for frag_id in local_fragments:
        l = frag_id * points_per_fragment
        r = min(numpoints, l + points_per_fragment)
        fragment = generate_fragment(r - l, dimensions, mode, seed + l)
        fragments_data.append(fragment)
    
    if rank == 0:
        print("Geração de dados concluída")
    
    # Loop principal do K-means
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        if rank == 0:
            print(f"Iteração #{iteration + 1}/{iterations}")
        
        old_centres = centres.copy()
        
        # Calcular somas parciais locais
        local_partials = []
        for fragment in fragments_data:
            partial = partial_sum(fragment, old_centres)
            local_partials.append(partial)
        
        # Agregar parciais locais
        if local_partials:
            accumulated_local = local_partials[0].copy()
            for partial in local_partials[1:]:
                accumulated_local += partial
        else:
            accumulated_local = np.zeros((centres.shape[0], 2), dtype=object)
            for i in range(centres.shape[0]):
                accumulated_local[i][0] = np.zeros(dimensions)
                accumulated_local[i][1] = 0
        
        # Redução global usando MPI
        # Preparar dados para envio
        sums_local = np.array([accumulated_local[i][0] for i in range(num_centres)])
        counts_local = np.array([accumulated_local[i][1] for i in range(num_centres)])
        
        sums_global = np.zeros_like(sums_local)
        counts_global = np.zeros_like(counts_local)
        
        comm.Allreduce(sums_local, sums_global, op=MPI.SUM)
        comm.Allreduce(counts_local, counts_global, op=MPI.SUM)
        
        # Recompute centres
        for idx in range(num_centres):
            if counts_global[idx] != 0:
                centres[idx] = sums_global[idx] / counts_global[idx]
        
        iteration += 1
    
    adios.finalize()
    
    return centres


def generate_fragment(points, dim, mode, seed):
    """
    Gera um fragmento aleatório de pontos
    
    :param points: Número de pontos
    :param dim: Número de dimensões
    :param mode: Modo de geração
    :param seed: Semente aleatória
    :return: Fragmento de dados
    """
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for __ in range(points)])
    
    # Normalizar pontos entre 0 e 1
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    
    return mat


def parse_arguments():
    """
    Processa argumentos da linha de comando
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="KMeans com ADIOS2")
    parser.add_argument(
        "-s", "--seed", type=int, default=0, 
        help="Semente pseudo-aleatória. Padrão = 0"
    )
    parser.add_argument(
        "-n", "--numpoints", type=int, default=100,
        help="Número de pontos. Padrão = 100"
    )
    parser.add_argument(
        "-d", "--dimensions", type=int, default=2,
        help="Número de dimensões. Padrão = 2"
    )
    parser.add_argument(
        "-c", "--num_centres", type=int, default=5,
        help="Número de centros. Padrão = 5"
    )
    parser.add_argument(
        "-f", "--fragments", type=int, default=10,
        help="Número de fragmentos. Padrão = 10"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="uniform",
        choices=["uniform", "normal"],
        help="Distribuição de pontos. Padrão = uniform"
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=20,
        help="Número máximo de iterações"
    )
    parser.add_argument(
        "-e", "--epsilon", type=float, default=1e-9,
        help="Epsilon. KMeans para quando: |old - new| < epsilon"
    )
    
    return parser.parse_args()


def main(seed, numpoints, dimensions, num_centres, fragments, mode, iterations, epsilon):
    """
    Função principal
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    start_time = time.time()
    
    if rank == 0:
        print("Iniciando K-means com ADIOS2")
    
    # Executar K-means
    centres = kmeans_adios(
        num_fragments=fragments,
        dimensions=dimensions,
        numpoints=numpoints,
        num_centres=num_centres,
        iterations=iterations,
        seed=seed,
        epsilon=epsilon,
        mode=mode,
    )
    
    end_time = time.time()
    
    if rank == 0:
        print("-----------------------------------------")
        print("-------------- RESULTADOS ---------------")
        print("-----------------------------------------")
        print(f"Tempo total: {end_time - start_time:.4f}s")
        print("-----------------------------------------")
        print("CENTROS:")
        print(centres)
        print("-----------------------------------------")


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))