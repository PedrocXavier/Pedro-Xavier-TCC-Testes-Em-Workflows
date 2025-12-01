import numpy as np
import time
import argparse
import sys

# Importações do Parsl
import parsl
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl import python_app
from parsl.dataflow.futures import Future

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

# ----------------------------------------------------------------------
# 🎯 Funções Decoradas Parsl (Tasks)
# ----------------------------------------------------------------------

@python_app
def partial_sum(fragment, centres):
    """
    Calcula a soma parcial dos pontos (fragment) e a contagem de pontos 
    para cada centro (centres). Retorna uma array NumPy.
    """
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        indices = np.argwhere(close_centres == center_idx).flatten()
        # partials[i][0] = soma dos pontos; partials[i][1] = contagem de pontos
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        partials[center_idx][1] = indices.shape[0]
    return partials


@python_app
def merge(*data):
    """
    Combina (reduz) uma lista de arrays parciais somando as somas e contagens.
    """
    accum = data[0].copy()
    for d in data[1:]:
        # Garante que a soma e a contagem de cada centro sejam somadas corretamente
        for i in range(accum.shape[0]):
            accum[i][0] = accum[i][0] + d[i][0] # Soma dos pontos
            accum[i][1] = accum[i][1] + d[i][1] # Contagem de pontos
    return accum


@python_app
def generate_fragment(points, dim, mode, seed):
    """
    Gera um fragmento aleatório do conjunto de dados.
    """
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for __ in range(points)])
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx
    return mat

# ----------------------------------------------------------------------
# ⚙️ Funções Auxiliares e Lógica Principal
# ----------------------------------------------------------------------

def converged(old_centres, centres, epsilon, iteration, max_iter):
    """
    Verifica se o algoritmo convergiu ou atingiu o número máximo de iterações.
    """
    if old_centres is None:
        return False
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


def recompute_centres(partials, old_centres, arity):
    """
    Recalcula os novos centros reduzindo as somas parciais.
    Espera pelo resultado final da redução usando .result().
    """
    centres = old_centres.copy()
    
    # Redução em árvore (Tree reduction) para combinar as somas parciais
    while len(partials) > 1:
        new_partials = []
        for i in range(0, len(partials), arity):
            partials_subset = partials[i:i + arity]
            # O resultado de merge é um novo Future
            new_partials.append(merge(*partials_subset))
        partials = new_partials

    # Espera pelo resultado final da redução (substitui compss_wait_on)
    final_partials_result = partials[0].result()

    # Recálculo do centro: Soma dos pontos / Contagem dos pontos
    for idx, sum_ in enumerate(final_partials_result):
        if sum_[1] != 0:
            centres[idx] = sum_[0] / sum_[1]
            
    return centres


def kmeans_frag(
    fragments,
    dimensions,
    num_centres=10,
    iterations=20,
    seed=0.0,
    epsilon=1e-9,
    arity=50,
):
    """
    Algoritmo K-Means baseado em fragmentos.
    :param fragments: Lista de arrays NumPy de dados (fragmentos)
    ...
    :return: Array NumPy final dos centros
    """
    np.random.seed(int(seed))
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None
    iteration = 0
    
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print("Doing iteration #%d/%d" % (iteration + 1, iterations))
        old_centres = centres.copy()
        partials = []
        
        # Inicia tarefas (tasks) paralelas para cada fragmento
        for frag in fragments:
            partial = partial_sum(frag, old_centres)
            partials.append(partial)
        
        # recompute_centres gerencia a redução em árvore e a espera (.result())
        centres = recompute_centres(partials, old_centres, arity)
        iteration += 1
    
    return centres


def parse_arguments():
    """
    Analisador de argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(description="KMeans Clustering.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Pseudo-random seed. Default = 0")
    parser.add_argument("-n", "--numpoints", type=int, default=100, help="Number of points. Default = 100")
    parser.add_argument("-d", "--dimensions", type=int, default=2, help="Number of dimensions. Default = 2")
    parser.add_argument("-c", "--num_centres", type=int, default=5, help="Number of centres. Default = 5")
    parser.add_argument("-f", "--fragments", type=int, default=10, help="Number of fragments. Default = 10. Condition: fragments < points")
    parser.add_argument("-m", "--mode", type=str, default="uniform", choices=["uniform", "normal"], help="Distribution of points. Default = uniform")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="Maximum number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9, help="Epsilon. KMeans will stop when: |old - new| < epsilon.")
    parser.add_argument("-a", "--arity", type=int, default=50, help="Arity of the reduction carried out during the computation of the new centroids")
    return parser.parse_args()


def setup_parsl_config(max_threads=8):
    """
    Cria e carrega a configuração do Parsl usando ThreadPoolExecutor.
    """
    config = Config(
        executors=[
            ThreadPoolExecutor(
                label="local_threads",
                max_threads=max_threads  
            )
        ]
    )
    parsl.load(config)
    print(f"✅ Configuração do Parsl carregada com ThreadPoolExecutor (max_threads={max_threads}).")
    

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
    """
    Função principal de execução.
    """
    start_time = time.time()

    # 1. Geração de Dados (Paralela)
    fragment_list = []
    points_per_fragment = max(1, numpoints // fragments)

    for l in range(0, numpoints, points_per_fragment):
        r = min(numpoints, l + points_per_fragment)
        # O resultado é um Future
        fragment_list.append(generate_fragment(r - l, dimensions, mode, seed + l))

    # Espera pelos resultados de geração (substitui compss_barrier)
    # Acessa os valores dos Futures para obter as arrays NumPy concretas
    concrete_fragments = [f.result() for f in fragment_list]
    
    print("Generation/Load done")
    initialization_time = time.time()
    print("Starting kmeans")

    # 2. Execução do K-Means (Iterativa e Paralela)
    centres = kmeans_frag(
        fragments=concrete_fragments, 
        dimensions=dimensions,
        num_centres=num_centres,
        iterations=iterations,
        seed=seed,
        epsilon=epsilon,
        arity=arity,
    )
    
    # O resultado de centres é a array NumPy final (já síncrona)
    print("Ending kmeans")
    kmeans_time = time.time()

    # 3. Impressão de Resultados
    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("KMeans time: %f" % (kmeans_time - initialization_time))
    print("Total time: %f" % (kmeans_time - start_time))
    print("-----------------------------------------")
    
    final_centres = centres
    print("CENTRES:")
    print(final_centres)
    print("-----------------------------------------")


if __name__ == "__main__":
    options = parse_arguments()
    
    # 🚨 Configuração do Parsl: ThreadPoolExecutor
    try:
        # Você pode ajustar o 'max_threads' aqui. Usando 8 como padrão.
        setup_parsl_config(max_threads=8)
    except Exception as e:
        print(f"Erro ao configurar o Parsl. Verifique se a biblioteca está instalada: {e}", file=sys.stderr)
        sys.exit(1)
        
    main(**vars(options))
    
    # 🧹 Descarrega o Parsl após a execução
    parsl.dfk().cleanup()