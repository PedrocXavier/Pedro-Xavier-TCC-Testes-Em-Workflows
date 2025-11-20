import numpy as np
import parsl
from parsl import python_app, bash_app
import math
import csv
import os

############################################
# CONFIGURAÇÃO DO PARSL
############################################
parsl.load(parsl.Config(
    executors=[
        parsl.executors.ThreadPoolExecutor(
            max_threads=4,  # Ajuste conforme máquina/cluster
            label="threads"
        )
    ]
))

############################################
# FUNÇÕES AUXILIARES
############################################

def momentum(u):
    return np.sum(u)

############################################
# TASK 1 – Inicialização
############################################
@python_app
def init_condition(N, dx):
    pi = math.pi
    u0 = np.zeros(N)
    for i in range(N):
        u0[i] = math.cos(pi * i * dx)
    return u0

############################################
# TASK 2 – Segunda etapa u1
############################################
@python_app
def compute_u1(N, dx, dt):
    pi = math.pi
    u1 = np.zeros(N)
    for i in range(N):
        cdt = math.cos(pi * i * dx) * dt
        u1[i] = math.cos(pi * (i * dx - cdt))
    return u1

############################################
# TASK 3 – Iteração principal
############################################
@python_app
def kdv_step(j, u0, u1, N, dx, dt, delta):
    pi = math.pi
    u0 = np.array(u0)
    u1 = np.array(u1)

    k1 = dt / (3 * dx)
    k2 = delta**2 * dt / (dx**3)

    u2 = np.zeros(N)
    
    # Borda 0
    t1 = (u1[1] + u1[0] + u1[N - 1]) * (u1[1] - u1[N - 1])
    t2 = u1[2] - 2 * u1[1] + 2 * u1[N - 1] - u1[N - 2]
    u2[0] = u0[0] - k1 * t1 - k2 * t2

    # Borda 1
    t1 = (u1[2] + u1[1] + u1[0]) * (u1[2] - u1[0])
    t2 = u1[3] - 2 * u1[2] + 2 * u1[0] - u1[N - 1]
    u2[1] = u0[1] - k1 * t1 - k2 * t2

    # Interior
    for i in range(2, N - 2):
        t1 = (u1[i+1] + u1[i] + u1[i-1]) * (u1[i+1] - u1[i-1])
        t2 = u1[i+2] - 2*u1[i+1] + 2*u1[i-1] - u1[i-2]
        u2[i] = u0[i] - k1*t1 - k2*t2

    # Borda N-2
    u2[N-2] = (
        u0[N-2]
        - k1*(u1[N-1]+u1[N-2]+u1[N-3])*(u1[N-1]-u1[N-3])
        - k2*(u1[0] - 2*u1[N-1] + 2*u1[N-3] - u1[N-4])
    )

    # Borda N-1
    u2[N-1] = (
        u0[N-1]
        - k1*(u1[0]+u1[N-1]+u1[N-2])*(u1[0]-u1[N-2])
        - k2*(u1[1] - 2*u1[0] + 2*u1[N-2] - u1[N-3])
    )

    # Verificação
    p = momentum(u2)
    if abs(p) > math.sqrt(np.finfo(float).eps):
        print(f"[WARN] Divergência possível no passo j={j}, momentum = {p}")

    return u2

############################################
# TASK 4 – Salvamento periódico
############################################
@python_app
def save_step(u, step):
    fname = f"kdv_output_step_{step}.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(u.tolist())
    return fname

############################################
# DRIVER PRINCIPAL – SIMULAÇÃO COMPLETA
############################################
def run_kdv(N=256, t_max=5.0, delta=0.022):
    dx = 2.0 / N
    dt = 27 * dx**3 / 4
    M = int(np.ceil(t_max / dt))

    skip_steps = 40000

    print(f"Parâmetros: N={N}, dt={dt}, t_max={t_max}, delta={delta}")

    # Inicialização
    u0_f = init_condition(N, dx)
    u1_f = compute_u1(N, dx, dt)

    # Loop temporal
    for j in range(1, M - 1):
        u2_f = kdv_step(j, u0_f, u1_f, N, dx, dt, delta)

        # Save only each skip_steps
        if (j+1) % skip_steps == 0:
            print(f"Salvando passo {j+1}/{M}")
            save_step(u2_f, j+1)

        # Atualiza dependências do Parsl
        u0_f = u1_f
        u1_f = u2_f

    print("Simulação concluída!")

############################################
# EXECUÇÃO
############################################
if __name__ == "__main__":
    run_kdv(N=256, t_max=5, delta=0.022)
