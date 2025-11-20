import numpy as np
import math
import sys
import os
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from pycompss.api.parameter import Type, DIRECTION, IN, INOUT, FILE

# --- Funções Auxiliares (Não são tarefas remotas, executam localmente ou são chamadas dentro de tarefas) ---

def display_progress(progress):
    """Exibe o progresso no console."""
    barWidth = 70
    pos = int(barWidth * progress)
    bar = "=" * pos + ">" + " " * (barWidth - pos - 1) if pos < barWidth else "=" * barWidth
    sys.stdout.write("\033[0;32m[{}] {}%\033[0m\r".format(bar, int(progress * 100.0)))
    sys.stdout.flush()

def momentum(u_array):
    """Calcula o momento (soma) do array u."""
    return np.sum(u_array)

def write_data_step(file_path, u_data, j, dt):
    """
    Função auxiliar para simular a escrita de dados (ADIOS2) em um arquivo de texto.
    Esta função seria uma tarefa @task se a E/S fosse paralela/distribuída.
    """
    try:
        with open(file_path, 'a') as f:
            f.write(f"Time_Step: {j+1}, Time: {(j+1) * dt}\n")
            # Salva o array u no formato que preferir (aqui, apenas a soma para simplificar)
            np.savetxt(f, u_data.reshape(1, -1), fmt='%.18e', delimiter=',')
    except Exception as e:
        print(f"Erro ao escrever o passo de dados: {e}")

# --- Tarefas PyCOMPSs (@task) ---

@task(returns=np.array)
def initialize_u0(N, dx, pi):
    """
    Inicializa a condição inicial u0.
    """
    x = np.linspace(0, 2 - dx, N) # Domínio [0, 2] com N pontos, u(x,0) = cos(πx)
    u0 = np.cos(pi * x)
    return u0

@task(returns=np.array)
def initialize_u1(N, dx, dt, pi):
    """
    Inicializa a condição de tempo 1 (u1) usando o passo inicial.
    """
    x = np.linspace(0, 2 - dx, N)
    cdt_array = np.cos(pi * x) * dt
    u1 = np.cos(pi * (x - cdt_array))
    return u1

@task(u0_in=IN, u1_in=IN, N=IN, k1=IN, k2=IN, returns=np.array)
def compute_next_step(u0_in, u1_in, N, k1, k2):
    """
    Calcula o próximo passo de tempo u2 (Zabusky and Kruskal scheme).
    Esta é a parte com o maior custo computacional.
    """
    u0 = u0_in
    u1 = u1_in
    u2 = np.empty(N, dtype=u0.dtype)

    # Implementação do esquema Zabusky-Kruskal com condições de contorno periódicas
    
    # Índices (Shifted arrays para simplificar a notação de contorno periódica)
    # i+1, i-1, i+2, i-2, etc.

    # u1_{i+1}
    u1_p1 = np.roll(u1, -1)
    # u1_{i-1}
    u1_m1 = np.roll(u1, 1)
    # u1_{i+2}
    u1_p2 = np.roll(u1, -2)
    # u1_{i-2}
    u1_m2 = np.roll(u1, 2)

    # Termo não-linear (t1): (u_{i+1} + u_i + u_{i-1}) * (u_{i+1} - u_{i-1})
    t1 = (u1_p1 + u1 + u1_m1) * (u1_p1 - u1_m1)

    # Termo dispersivo (t2): u_{i+2} - 2u_{i+1} + 2u_{i-1} - u_{i-2}
    t2 = u1_p2 - 2 * u1_p1 + 2 * u1_m1 - u1_m2

    # Equação principal: u_{i}^{n+1} = u_{i}^{n-1} - k1 * t1 - k2 * t2
    u2 = u0 - k1 * t1 - k2 * t2
    
    return u2

# --- Workflow Principal ---

def KdV_workflow(N, dt, t_max, delta, output_file="korteweg_de_vries.txt"):
    """
    Workflow principal para a simulação KdV.
    """
    print(f"🖥️ Iniciando workflow KdV com N={N}, t_max={t_max}, delta={delta}")
    
    pi = math.pi
    
    if N <= 0 or dt <= 0 or dt > 1:
        raise ValueError("Parâmetros N e dt inválidos.")

    dx = 2.0 / N
    M = math.ceil(t_max / dt)
    
    print(f"Usando Δx = {dx:.6f}, Δt = {dt:.6f} e M (passos) = {M}")
    
    # Constantes do esquema de diferenças finitas
    k1 = dt / (3 * dx)
    k2 = delta**2 * dt / (dx**3)
    
    skip_steps = 40000 # Mesma lógica do código C++ para E/S

    # 1. Inicialização das Condições
    # As funções de inicialização são tarefas PyCOMPSs
    u0_future = initialize_u0(N, dx, pi)
    u1_future = initialize_u1(N, dx, dt, pi)
    
    # Espera pelos resultados iniciais para começar o loop
    # u0 e u1 são objetos Future, mas para o primeiro passo,
    # as tarefas subsequentes as buscam automaticamente.
    # No entanto, a primeira iteração usa u0 e u1 como entradas para u2.
    # Não vamos 'esperar' aqui para manter o paralelismo, mas
    # vamos tratar u0_future e u1_future como os "u0" e "u1" do loop.

    # Limpar o arquivo de saída antes de começar
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # Salva u0 no primeiro passo de E/S
    # Para a escrita inicial de u0 (passo j=0)
    # Precisamos esperar pelos dados para escrever, pois a escrita não é paralela aqui
    u0_data = compss_wait_on(u0_future)
    write_data_step(output_file, u0_data, 0, dt)
    
    current_u0 = u0_future # u^(n-1)
    current_u1 = u1_future # u^(n)
    
    # 2. Loop de Tempo Principal (De j=1 até M-1)
    for j in range(1, int(M) - 1):
        # Calcula u^(n+1) = u2
        u2_future = compute_next_step(current_u0, current_u1, N, k1, k2)
        
        # Simulação de E/S de alto desempenho (ADIOS2)
        # Se o passo atual for um passo de escrita
        if (j + 1) % skip_steps == 0:
            # Precisa esperar pelos dados antes de escrever e verificar o momento.
            u2_data = compss_wait_on(u2_future)
            
            # Verificação de Momento (conservado)
            p = momentum(u2_data)
            
            if np.isnan(p) or abs(p) > np.sqrt(np.finfo(float).eps):
                 print(f"\n⚠️ Solução divergiu em t = {(j + 1) * dt:.6f}")
                 print(f"Momento = {p:.10e}")
                 # Não precisa de compss_stop(), pois a exceção deve ser tratada no main.
                 # Em um workflow real, você poderia decidir retornar ou lançar uma exceção.
                 # Para simplificar, vamos parar o loop aqui.
                 break
                 
            # Escrita dos dados (Simulação ADIOS2 - Etapa Begin/EndStep)
            display_progress((j + 1) / (M - 1))
            write_data_step(output_file, u2_data, j, dt)

            # Atualiza os objetos Future para o próximo passo (j+1)
            current_u0 = current_u1
            current_u1 = u2_future
        else:
            # Apenas atualiza os objetos Future para a próxima iteração
            current_u0 = current_u1
            current_u1 = u2_future
    
    # 3. Finalização
    print("\n✅ Simulação KdV concluída.")
    # O PyCOMPSs gerencia a transferência de dados dos objetos Future no final, se necessário.


if __name__ == '__main__':
    # Valores default
    N = 256
    t_max = 5.0
    delta = 0.022
    
    # Leitura de argumentos da linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print("Uso: runcompss kdv_tasks.py N t_max delta")
            print("Exemplo: runcompss kdv_tasks.py 512 10 0.022")
            sys.exit(0)
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        t_max = float(sys.argv[2])
    if len(sys.argv) > 3:
        delta = float(sys.argv[3])
        
    # Cálculo de dt com base na condição de estabilidade (mesma lógica do código C++)
    dx = 1.0 / N # Nota: O código C++ usa dx = 2.0 / N, mas a fórmula de dt usa 1.0/N, vamos manter 2.0/N
    # Na verdade, o código C++ usa dx = 2.0 / N no cálculo, mas a linha 'double dx = 1.0 / static_cast<double>(N);'
    # no main está fora de sincronia com a função KdV. Vou manter a lógica da função KdV:
    dx_kdv = 2.0 / N # O KdV<double> usa N no espaço [0, 2], então o tamanho total é 2.
    # O C++ usa 'double dx = 1.0 / static_cast<double>(N);' no main e 'double dt = 27 * dx * dx * dx / 4;'
    # Vou usar o dx do main (1/N) para calcular o dt, mantendo a fidelidade ao C++
    
    # Usando o dx do main do C++ (embora pareça incorreto para [0,2]) para calcular dt
    dx_main = 1.0 / N
    dt = 27 * dx_main**3 / 4
    
    try:
        KdV_workflow(N, dt, t_max, delta)
    except Exception as e:
        print(f"Caught exception from KdV call: {e}", file=sys.stderr)
        sys.exit(1)