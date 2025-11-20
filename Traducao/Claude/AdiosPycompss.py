"""
Distributed under the OSI-approved Apache License, Version 2.0.

Solves the initial value problem for the Korteweg de-Vries equation via the
Zabusky and Krustal scheme using Parsl for workflow orchestration.

See: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.15.240 
Zabusky, Norman J., and Martin D. Kruskal. "Interaction of solitons in a 
collisionless plasma and the recurrence of initial states." 
Physical review letters 15.6 (1965): 240.
"""

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
import numpy as np
import h5py
import sys
from typing import List, Tuple
import os


def display_progress(progress: float):
    """Exibe barra de progresso no terminal."""
    bar_width = 70
    print("\033[0;32m[", end="")
    pos = int(bar_width * progress)
    for i in range(bar_width):
        if i < pos:
            print("=", end="")
        elif i == pos:
            print(">", end="")
        else:
            print(" ", end="")
    print(f"] {int(progress * 100.0)}%\033[0m\r", end="", flush=True)


def momentum(u: np.ndarray) -> float:
    """
    Calcula o momentum, uma quantidade conservada do método numérico.
    Usado para verificar a validade da solução.
    """
    return np.sum(u)


@python_app
def initialize_solution(N: int, dx: float) -> np.ndarray:
    """
    App Parsl para inicializar a condição inicial.
    u(x,0) = cos(πx) para x ∈ [0,2]
    """
    import numpy as np
    pi = np.pi
    u0 = np.zeros(N)
    for i in range(N):
        u0[i] = np.cos(pi * i * dx)
    return u0


@python_app
def compute_first_step(u0: np.ndarray, N: int, dx: float, dt: float) -> np.ndarray:
    """
    App Parsl para calcular o primeiro passo temporal.
    """
    import numpy as np
    pi = np.pi
    u1 = np.zeros(N)
    for i in range(N):
        cdt = np.cos(pi * i * dx) * dt
        u1[i] = np.cos(pi * (i * dx - cdt))
    return u1


@python_app
def compute_time_step(u0: np.ndarray, u1: np.ndarray, N: int, 
                      k1: float, k2: float, j: int) -> Tuple[np.ndarray, float, int]:
    """
    App Parsl para calcular um passo temporal da equação KdV.
    Retorna: (u2, momentum, step_number)
    """
    import numpy as np
    
    u2 = np.zeros(N)
    
    # Índice 0 (boundary condition)
    t1 = (u1[1] + u1[0] + u1[N-1]) * (u1[1] - u1[N-1])
    t2 = u1[2] - 2*u1[1] + 2*u1[N-1] - u1[N-2]
    u2[0] = u0[0] - k1*t1 - k2*t2
    
    # Índice 1
    t1 = (u1[2] + u1[1] + u1[0]) * (u1[2] - u1[0])
    t2 = u1[3] - 2*u1[2] + 2*u1[0] - u1[N-1]
    u2[1] = u0[1] - k1*t1 - k2*t2
    
    # Índices internos
    for i in range(2, N-2):
        t1 = (u1[i+1] + u1[i] + u1[i-1]) * (u1[i+1] - u1[i-1])
        t2 = u1[i+2] - 2*u1[i+1] + 2*u1[i-1] - u1[i-2]
        u2[i] = u0[i] - k1*t1 - k2*t2
    
    # Índice N-2
    t1 = (u1[N-1] + u1[N-2] + u1[N-3]) * (u1[N-1] - u1[N-3])
    t2 = u1[0] - 2*u1[N-1] + 2*u1[N-3] - u1[N-4]
    u2[N-2] = u0[N-2] - k1*t1 - k2*t2
    
    # Índice N-1 (boundary condition)
    t1 = (u1[0] + u1[N-1] + u1[N-2]) * (u1[0] - u1[N-2])
    t2 = u1[1] - 2*u1[0] + 2*u1[N-2] - u1[N-3]
    u2[N-1] = u0[N-1] - k1*t1 - k2*t2
    
    # Calcula momentum
    p = np.sum(u2)
    
    return u2, p, j


@python_app
def save_to_hdf5(filename: str, u_data: np.ndarray, step: int, 
                 dx: float, dt: float, append: bool = True) -> bool:
    """
    App Parsl para salvar dados em formato HDF5.
    """
    import h5py
    import numpy as np
    
    mode = 'a' if append else 'w'
    with h5py.File(filename, mode) as f:
        if not append:
            # Criar dataset com chunks para eficiência
            f.create_dataset('u', data=u_data[np.newaxis, :], 
                           maxshape=(None, u_data.shape[0]), 
                           chunks=True, compression='gzip')
            f.attrs['x0'] = 0.0
            f.attrs['dx'] = dx
            f.attrs['dt'] = dt
            f.attrs['interpretation'] = 'Equispaced'
        else:
            # Adicionar novo timestep
            dset = f['u']
            dset.resize((dset.shape[0] + 1, dset.shape[1]))
            dset[-1, :] = u_data
    
    return True


def KdV_parsl(N: int, dt: float, t_max: float, delta: float = 0.022, 
              output_file: str = "korteweg_de_vries.h5"):
    """
    Resolve o problema de valor inicial para a equação KdV usando Parsl.
    
    Parâmetros:
    - N: número de pontos espaciais
    - dt: passo temporal
    - t_max: tempo máximo de simulação
    - delta: parâmetro de interação (δ)
    - output_file: arquivo de saída HDF5
    """
    
    if N <= 0:
        raise ValueError("N > 0 é necessário")
    if dt > 1:
        raise ValueError("Passo temporal muito grande")
    if dt <= 0:
        raise ValueError("dt > 0 é necessário")
    
    dx = 2.0 / N
    M = int(np.ceil(t_max / dt))
    
    print(f"Resolvendo o problema de valor inicial para a equação KdV "
          f"∂tu + u∂ₓu + δ²∂ₓ³u = 0 usando δ = {delta}.")
    print(f"Condições iniciais: u(x,0) = cos(πx) para x ∈ [0,2].")
    print(f"Usando ∆x = {dx}, ∆t = {dt} e t_max = {t_max}")
    print(f"Total de passos temporais: {M}\n")
    
    # Inicializar solução
    u0_future = initialize_solution(N, dx)
    u0 = u0_future.result()
    
    # Salvar condição inicial
    save_to_hdf5(output_file, u0, 0, dx, dt, append=False).result()
    
    # Primeiro passo temporal
    u1_future = compute_first_step(u0, N, dx, dt)
    u1 = u1_future.result()
    
    # Constantes do esquema numérico
    k1 = dt / (3 * dx)
    k2 = delta * delta * dt / (dx**3)
    
    skip_steps = 40000  # Pular passos para não escrever dados demais
    
    # Loop principal de integração temporal
    for j in range(1, M-1):
        # Computar próximo passo temporal
        future = compute_time_step(u0, u1, N, k1, k2, j)
        u2, p, step = future.result()
        
        # Verificar convergência
        if np.isnan(p):
            print(f"\nSolução divergiu em t = {(j+1)*dt}")
            print(f"Momentum = {p}")
            return
        
        if np.abs(p) > np.sqrt(np.finfo(float).eps):
            print(f"\nAtenção: Momentum não está conservado em t = {(j+1)*dt}")
            print(f"Momentum = {p}")
        
        # Salvar dados periodicamente
        if (j + 1) % skip_steps == 0:
            display_progress((j + 1) / (M - 1))
            save_to_hdf5(output_file, u2, j+1, dx, dt, append=True).result()
        
        # Atualizar para próxima iteração
        u0 = u1
        u1 = u2
    
    print("\n\nSimulação concluída!")
    print(f"Dados salvos em: {output_file}")


def main():
    """Função principal para executar a simulação."""
    
    # Configuração do Parsl
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="local_htex",
                worker_debug=False,
                cores_per_worker=1,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        strategy='none',
    )
    
    # Inicializar Parsl
    parsl.load(config)
    
    # Parâmetros padrão
    N = 256
    t_max = 5.0
    delta = 0.022
    
    # Processar argumentos da linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Uso: python kdv_parsl.py N t_max δ")
            print("  N: número de pontos espaciais (∆t escolhido via condição de estabilidade)")
            print("  t_max: tempo máximo de simulação")
            print("  δ: parâmetro de interação")
            print("Exemplo: python kdv_parsl.py 512 10 0.022")
            parsl.clear()
            return 0
        N = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        t_max = float(sys.argv[2])
    
    if len(sys.argv) > 3:
        delta = float(sys.argv[3])
    
    # Calcular dt pela condição de estabilidade
    dx = 1.0 / N
    dt = 27 * dx**3 / 4
    
    try:
        KdV_parsl(N, dt, t_max, delta)
    except Exception as e:
        print(f"Exceção capturada durante execução do KdV: {e}")
        parsl.clear()
        return 1
    
    # Finalizar Parsl
    parsl.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())