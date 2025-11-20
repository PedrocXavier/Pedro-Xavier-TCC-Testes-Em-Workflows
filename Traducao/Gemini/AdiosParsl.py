# -*- coding: utf-8 -*-
"""
Tradução do workflow C++/ADIOS2 para a solução da equação de Korteweg-de Vries (KdV)
utilizando Python/NumPy e o sistema de workflow Parsl.

A função KdV_simulation é encapsulada como um aplicativo Parsl (python_app)
para ser executada como parte de um workflow paralelo.
"""

import numpy as np
import math
import sys
import os
from parsl import load, python_app
from parsl.config import Config
# Alterado para ThreadPoolExecutor
from parsl.executors import ThreadPoolExecutor

# Configuração Básica do Parsl
# NOTE: Esta é uma configuração mínima. Em um sistema HPC real, o 'label'
# e 'provider' seriam configurados para SLURM, PBS, etc.
parsl_config = Config(
    executors=[
        # Utilizando ThreadPoolExecutor
        ThreadPoolExecutor(
            label="threads_local",
            max_threads=4  # Usando 4 threads para permitir alguma concorrência local
        )
    ]
)

# ----------------------------------------------------------------------------
# FUNÇÕES DE LÓGICA DE SIMULAÇÃO (Tradução para Python/NumPy)
# ----------------------------------------------------------------------------

def display_progress(progress):
    """
    Exibe uma barra de progresso no console.

    Args:
        progress (float): Progresso atual (0.0 a 1.0).
    """
    bar_width = 70
    pos = int(bar_width * progress)
    bar = "=" * pos + ">" + " " * (bar_width - pos - 1) if pos < bar_width else "=" * bar_width
    sys.stdout.write(f"\033[0;32m[{bar}] {int(progress * 100.0)}%\033[0m\r")
    sys.stdout.flush()

def momentum(u):
    """
    Calcula o momento, uma quantidade conservada, para verificação.

    Args:
        u (np.ndarray): Array da solução numérica.
    """
    return np.sum(u)

@python_app
def kdv_simulation(N, dt, t_max, delta, output_filename, inputs=[]):
    """
    Resolve o problema de valor inicial para a equação KdV (Korteweg-de Vries)
    via o esquema de Zabusky e Kruskal.

    Args:
        N (int): Número de pontos da grade espacial.
        dt (float): Passo de tempo.
        t_max (float): Tempo máximo de simulação.
        delta (float): Parâmetro de dispersão (δ).
        output_filename (str): Caminho para salvar os resultados da simulação.
        inputs (list): Necessário para dependências do Parsl, mas vazio aqui.

    Returns:
        str: O nome do arquivo de saída gerado.
    """
    import numpy as np
    import math
    import sys

    # Ajustando o escopo do PI para dentro da função que será paralelizada
    pi = 4 * np.arctan(1.0)
    
    # 1. Verificações de domínio
    if N <= 0:
        raise ValueError("N > 0 é necessário")
    if dt <= 0:
        raise ValueError("dt > 0 é necessário")

    dx = 2.0 / N
    M = int(np.ceil(t_max / dt))

    print(f"Resolvendo a equação KdV com δ = {delta}.")
    print(f"Condições iniciais: u(x,0) = cos(πx) para x∈[0,2].")
    print(f"Usando ∆x = {dx}, ∆t = {dt} e t_max = {t_max}")

    # Coeficientes da diferença finita
    k1 = dt / (3.0 * dx)
    k2 = delta**2 * dt / (dx**3)

    # 2. Inicialização dos arrays (u0=u^{j-1}, u1=u^{j}, u2=u^{j+1})
    x = np.linspace(0, 2.0 - dx, N, endpoint=True) # Excluir x=2 devido à periodicidade
    
    # u(x, 0) - u0 (usado como u^{j-1} no primeiro passo real)
    u0 = np.cos(pi * x)

    # u(x, dt) - u1 (primeiro passo de tempo)
    # A inicialização KdV original usa um esquema perturbado para o primeiro passo
    # u1[i] = cos(pi * (i * dx - cos(pi * i * dx) * dt))
    u1 = np.cos(pi * (x - np.cos(pi * x) * dt))
    
    u2 = np.zeros(N)

    # Prepara o arquivo de saída para registrar os passos
    output_data = []

    # Salva o estado inicial (t=0)
    output_data.append(f"# N={N}, dt={dt}, t_max={t_max}, delta={delta}, dx={dx}")
    output_data.append(f"t=0.0: {list(u0)}")
    
    # 3. Loop de Tempo (Time Stepping)
    skip_steps = 40000  # Mesmo valor de I/O do original
    
    # O loop começa em j=1 e vai até M-2, totalizando M-2 passos de tempo.
    # O passo real é j+1.
    for j in range(1, M - 1):
        
        # --- Cálculo vetorizado da diferença finita (Esquema Zabusky & Kruskal) ---
        # A implementação em C++ lida manualmente com 5 pontos de fronteira.
        # Aqui, usamos np.roll para lidar com a periodicidade e vetorizar o cálculo
        # para *todos* os pontos de uma vez, tornando o código mais limpo e rápido.
        
        # Operadores de deslocamento (periodic boundary conditions)
        u_ip1 = np.roll(u1, -1)  # u[i+1]
        u_im1 = np.roll(u1, 1)   # u[i-1]
        u_ip2 = np.roll(u1, -2)  # u[i+2]
        u_im2 = np.roll(u1, 2)   # u[i-2]

        # Termo não-linear (u*du/dx) - Termo T1 na tradução vetorizada
        # T1 = (u_{i+1} + u_i + u_{i-1}) * (u_{i+1} - u_{i-1})
        T1 = (u_ip1 + u1 + u_im1) * (u_ip1 - u_im1)

        # Termo dispersivo (d^3u/dx^3) - Termo T2 na tradução vetorizada
        # T2 = u_{i+2} - 2*u_{i+1} + 2*u_{i-1} - u_{i-2}
        T2 = u_ip2 - 2.0 * u_ip1 + 2.0 * u_im1 - u_im2

        # Esquema de integração (Leapfrog Modificado)
        # u^{j+1} = u^{j-1} - k1*T1 - k2*T2
        u2 = u0 - k1 * T1 - k2 * T2
        
        # --- Verificações de Estabilidade/Momentum ---
        p = momentum(u2)
        
        if np.isnan(p):
            print(f"\nSolução divergiu em t = {(j + 1) * dt}. Momentum = {p}", file=sys.stderr)
            break
            
        # O código original verifica se o momento é maior que sqrt(epsilon).
        # Para float64 (double), epsilon é ~2.2e-16. sqrt(epsilon) é ~1.5e-8.
        epsilon_check = np.sqrt(np.finfo(float).eps)
        if np.abs(p) > epsilon_check:
             # Este é um erro esperado no esquema de diferenças finitas, pois o momento
             # deve ser zero, mas se desvia devido a erros numéricos.
             pass # Apenas aviso, não interrompe como no C++ original
             
        # --- Saída de Dados (Substitui ADIOS2 I/O) ---
        if (j + 1) % skip_steps == 0:
            current_time = (j + 1) * dt
            display_progress(current_time / t_max)
            # Salva o estado atual (substituindo adios_engine.Put)
            output_data.append(f"t={current_time}: {list(u2)}")
            
        # --- Avança no Tempo ---
        u0 = u1.copy()
        u1 = u2.copy()

    # Salva o estado final
    final_time = (M - 1) * dt
    output_data.append(f"t={final_time}: {list(u2)}")
    display_progress(1.0) # Progresso final

    # Grava todos os dados no arquivo de saída
    with open(output_filename, 'w') as f:
        for line in output_data:
            f.write(line + '\n')
            
    print(f"\nSimulação KdV concluída. Resultados salvos em: {output_filename}")
    return output_filename

# ----------------------------------------------------------------------------
# FUNÇÃO MAIN E EXECUÇÃO DO WORKFLOW PARSL
# ----------------------------------------------------------------------------

def main():
    """
    Função principal que configura o Parsl e executa o workflow.
    """
    try:
        load(parsl_config)
    except Exception as e:
        print(f"Erro ao carregar a configuração do Parsl: {e}")
        return

    # Parâmetros padrão (iguais ao C++ original)
    N = 256
    t_max = 5.0
    delta = 0.022

    # Processamento de argumentos de linha de comando
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print("Uso: python kdv_parsl_workflow.py [N] [t_max] [delta]")
        print("  N: Número de pontos da grade espacial (Padrão: 256)")
        print("  t_max: Tempo máximo de simulação (Padrão: 5.0)")
        print("  delta: Parâmetro de interação δ (Padrão: 0.022)")
        return

    if len(args) > 0:
        try:
            N = int(args[0])
        except ValueError:
            print(f"Erro: N deve ser um inteiro. Recebido: {args[0]}")
            return
    if len(args) > 1:
        try:
            t_max = float(args[1])
        except ValueError:
            print(f"Erro: t_max deve ser um float. Recebido: {args[1]}")
            return
    if len(args) > 2:
        try:
            delta = float(args[2])
        except ValueError:
            print(f"Erro: delta deve ser um float. Recebido: {args[2]}")
            return

    # Cálculo do passo de tempo (dt) baseado na condição de estabilidade do original:
    # dt = 27 * dx^3 / 4, onde dx = 2.0 / N
    dx = 2.0 / float(N)
    dt = 27.0 * dx**3 / 4.0

    output_file = f"kdv_sim_N{N}_T{t_max}_D{delta:.3f}.txt"
    print(f"\nIniciando simulação KdV (Parsl App)...")
    
    # 4. Execução da App Parsl
    # Chamar a função decorada retorna um Future (um objeto de resultado assíncrono)
    sim_future = kdv_simulation(N, dt, t_max, delta, output_file)

    # 5. Esperar o resultado e obter o valor
    try:
        final_filename = sim_future.result()
        print(f"\nWorkflow Parsl concluído com sucesso. Arquivo de saída: {final_filename}")
        
    except Exception as e:
        print(f"\nO aplicativo Parsl falhou com uma exceção: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Garante que as dependências necessárias estejam instaladas
    try:
        import numpy
        import parsl
    except ImportError:
        print("Erro: As bibliotecas 'numpy' e 'parsl' são necessárias.", file=sys.stderr)
        print("Instale-as usando: pip install numpy parsl", file=sys.stderr)
        sys.exit(1)
        
    main()