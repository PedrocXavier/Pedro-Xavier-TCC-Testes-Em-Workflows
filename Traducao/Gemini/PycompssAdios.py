import numpy as np
import time
import adios2 # Importa a biblioteca ADIOS2

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances

# --- As funções de PyCOMPSs são convertidas em funções Python regulares ---

# A função partial_sum agora é executada diretamente, não como uma task
def partial_sum(fragment, centres):
    """
    Calcula a soma parcial e contagem de pontos para cada centro em um fragmento.
    """
    # 
    partials = np.zeros((centres.shape[0], 2), dtype=object)
    # Encontra o centro mais próximo para cada ponto do fragmento
    close_centres = pairwise_distances(fragment, centres).argmin(axis=1)
    for center_idx, _ in enumerate(centres):
        # Encontra os índices de pontos atribuídos ao centro 'center_idx'
        indices = np.argwhere(close_centres == center_idx).flatten()
        # Soma dos pontos
        partials[center_idx][0] = np.sum(fragment[indices], axis=0)
        # Contagem de pontos
        partials[center_idx][1] = indices.shape[0]
    return partials


# A função merge agora é uma função de agregação serial
def merge(*data):
    """
    Agrega as somas parciais.
    """
    accum = data[0].copy()
    for d in data[1:]:
        # O acumulador é um array de arrays onde a primeira coluna é a soma 
        # e a segunda é a contagem.
        accum[:, 0] += d[:, 0]
        accum[:, 1] += d[:, 1]
    return accum


def converged(old_centres, centres, epsilon, iteration, max_iter):
    """
    Verifica a convergência.
    """
    if old_centres is None:
        return False
    # Paired_distances retorna o array de distâncias euclidianas por par.
    # O somatório verifica a mudança total.
    dist = np.sum(paired_distances(centres, old_centres))
    return dist < epsilon**2 or iteration >= max_iter


def recompute_centres(partials, old_centres, arity):
    """
    Recomputa os centros após a agregação de todas as parciais.
    """
    centres = old_centres.copy()
    
    # Executa a redução (merge) de forma serial
    while len(partials) > 1:
        partials_subset = partials[:arity]
        partials = partials[arity:]
        
        # Merge de subconjuntos de parciais
        merged_partial = merge(*partials_subset)
        partials.append(merged_partial)

    # Não há necessidade de compss_wait_on, o resultado está pronto.
    final_partials = partials[0]

    # Recomputa os centros
    for idx, sum_ in enumerate(final_partials):
        if sum_[1] != 0: # sum_[1] é a contagem de pontos
            centres[idx] = sum_[0] / sum_[1] # sum_[0] é a soma dos pontos
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
    Algoritmo K-Means fragmentado (serializado).
    :param fragments: Lista de fragmentos de dados (arrays NumPy)
    ...
    :return: Centros finais
    """
    # Set the random seed
    np.random.seed(seed)
    
    # Inicialização dos centros
    centres = np.asarray([np.random.random(dimensions) for _ in range(num_centres)])
    old_centres = None
    iteration = 0
    
    while not converged(old_centres, centres, epsilon, iteration, iterations):
        print("Doing iteration #%d/%d" % (iteration + 1, iterations))
        old_centres = centres.copy()
        partials = []
        
        # Processamento serial de fragmentos
        for frag in fragments:
            # partial_sum é chamada diretamente
            partial = partial_sum(frag, old_centres)
            partials.append(partial)
            
        # Recomputação dos centros
        centres = recompute_centres(partials, old_centres, arity)
        iteration += 1
        
    return centres


def parse_arguments():
    """
    Analisa os argumentos da linha de comando.
    """
    import argparse
    # ... (o conteúdo da função parse_arguments permanece o mesmo)
    parser = argparse.ArgumentParser(description="KMeans Clustering.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Pseudo-random seed. Default = 0")
    parser.add_argument("-n", "--numpoints", type=int, default=100, help="Number of points. Default = 100")
    parser.add_argument("-d", "--dimensions", type=int, default=2, help="Number of dimensions. Default = 2")
    parser.add_argument("-c", "--num_centres", type=int, default=5, help="Number of centres. Default = 2")
    parser.add_argument("-f", "--fragments", type=int, default=10, help="Number of fragments." + " Default = 10. Condition: fragments < points")
    parser.add_argument("-m", "--mode", type=str, default="uniform", choices=["uniform", "normal"], help="Distribution of points. Default = uniform")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="Maximum number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9, help="Epsilon. KMeans will stop when:" + " |old - new| < epsilon.",)
    parser.add_argument("-a", "--arity", type=int, default=50, help="Arity of the reduction carried out during the computation of the new centroids")
    return parser.parse_args()


# --- Remoção da @task, agora é uma função auxiliar para gerar dados ---
def generate_fragment(points, dim, mode, seed):
    """
    Gera um fragmento aleatório de dados (serial).
    """
    # Random generation distributions
    rand = {
        "normal": lambda k: np.random.normal(0, 1, k),
        "uniform": lambda k: np.random.random(k),
    }
    r = rand[mode]
    np.random.seed(seed)
    mat = np.asarray([r(dim) for __ in range(points)])
    # Normalize all points between 0 and 1
    mat -= np.min(mat)
    mx = np.max(mat)
    if mx > 0.0:
        mat /= mx

    return mat


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
    
    # Variáveis de I/O para ADIOS2
    adios = adios2.Adios()
    io = adios.declare_io("KMeansData")
    # Define o motor de I/O (por exemplo, BP4)
    io.set_engine("BP4") 

    # --- Geração e Escrita dos dados usando ADIOS2 ---
    fragment_list = []
    points_per_fragment = max(1, numpoints // fragments)
    
    print("Iniciando Geração e Escrita de fragmentos com ADIOS2...")

    # Abre o motor para escrita
    with io.open("kmeans.bp", adios2.Mode.Write) as writer:
        for l in range(0, numpoints, points_per_fragment):
            r = min(numpoints, l + points_per_fragment)
            fragment_data = generate_fragment(r - l, dimensions, mode, seed + l)
            
            # Escreve o fragmento usando o ADIOS2
            var_name = f"Fragment_{l//points_per_fragment}"
            var = io.define_variable(var_name, fragment_data.astype(np.float64), 
                                    fragment_data.shape, [0]*len(fragment_data.shape), 
                                    fragment_data.shape, adios2.constant_dims)
            writer.Put(var, fragment_data, adios2.Mode.Sync)
            
            # Para o processamento serial, mantemos o fragmento na memória
            fragment_list.append(fragment_data) 

    # Não há necessidade de compss_barrier(), pois o I/O serial é síncrono.
    print("Geração/Escrita com ADIOS2 e Carregamento (em memória) concluídos")
    initialization_time = time.time()
    print("Starting kmeans")

    # Run kmeans
    centres = kmeans_frag(
        fragments=fragment_list,
        dimensions=dimensions,
        num_centres=num_centres,
        iterations=iterations,
        seed=seed,
        epsilon=epsilon,
        arity=arity,
    )
    
    # Não há necessidade de compss_barrier() ou compss_wait_on()
    print("Ending kmeans")
    kmeans_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("KMeans time: %f" % (kmeans_time - initialization_time))
    print("Total time: %f" % (kmeans_time - start_time))
    print("-----------------------------------------")
    # O resultado já é o array NumPy final, não um futuro do COMPSs
    print("CENTRES:")
    print(centres)
    print("-----------------------------------------")


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))