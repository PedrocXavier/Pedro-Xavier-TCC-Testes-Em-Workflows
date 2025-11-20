import numpy as np
from math import atan, cos, sqrt
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
import os


# ============================================================
# Helper: momentum
# ============================================================

def momentum(u):
    return np.sum(u)


# ============================================================
# Parallel domain decomposition utilities
# ============================================================

def split_domain(u, num_blocks):
    """Divide u em blocos contíguos para paralelização."""
    N = len(u)
    block_size = N // num_blocks
    blocks = []

    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < num_blocks - 1 else N
        blocks.append(u[start:end].copy())

    return blocks


def join_blocks(blocks):
    """Reconstrói o vetor completo."""
    return np.concatenate(blocks)


# ============================================================
# Tarefas PyCOMPSs
# ============================================================

@task(returns=np.ndarray)
def compute_block(u0_blk, u1_blk, left_ghost, right_ghost, dx, dt, delta):
    """
    Computa um bloco da solução do KdV usando vizinhos (ghost cells).
    """
    Nblk = len(u0_blk)
    u2 = np.zeros(Nblk)

    k1 = dt / (3 * dx)
    k2 = delta * delta * dt / (dx * dx * dx)

    # Construir vetor estendido com células fantasma
    extended = np.concatenate(([left_ghost], u1_blk, [right_ghost]))

    # Loop interno (esquema compacto usando extended[])
    for i in range(Nblk):
        e = extended  # alias
        t1 = (e[i+2] + e[i+1] + e[i]) * (e[i+2] - e[i])
        t2 = e[i+3] - 2*e[i+2] + 2*e[i] - e[i-1]
        u2[i] = u0_blk[i] - k1*t1 - k2*t2

    return u2


@task()
def write_snapshot_csv(u, dx, step, outdir):
    """Escreve snapshot em CSV estruturado e legível."""
    N = len(u)
    x_vals = np.arange(N) * dx

    data = np.column_stack([np.arange(N), x_vals, u])
    header = "index,x,u_value"

    file = os.path.join(outdir, f"snapshot_{step}.csv")
    np.savetxt(file, data, delimiter=",", header=header, comments="", fmt="%.8f")

    print(f"[PyCOMPSs] Snapshot salvo em {file}")


# ============================================================
# Programa Principal (paralelo)
# ============================================================

def KdV_parallel(N=512, t_max=5.0, delta=0.022, num_blocks=8, outdir="kdv_parallel_out"):

    os.makedirs(outdir, exist_ok=True)

    dx = 1.0 / N
    dt = 27 * dx * dx * dx / 4
    M = int(np.ceil(t_max / dt))
    skip_steps = 10000  # aumentar a frequência das saídas em paralelo

    print(f"Iniciando simulação paralela KdV com {num_blocks} blocos…")

    # Condições iniciais
    pi = 4 * atan(1.0)
    u0 = np.array([cos(pi * i * dx) for i in range(N)])
    u1 = np.array([cos(pi * (i * dx - cos(pi * i * dx) * dt)) for i in range(N)])

    # Split inicial
    u0_blocks = split_domain(u0, num_blocks)
    u1_blocks = split_domain(u1, num_blocks)

    # ----------------------------------------------------------
    # LOOP DE TEMPO PARALELO
    # ----------------------------------------------------------
    for step in range(1, M - 1):

        next_blocks = []

        for b in range(num_blocks):

            u0_blk = u0_blocks[b]
            u1_blk = u1_blocks[b]

            # Ghost cells
            left = u1_blocks[b-1][-1] if b > 0 else u1_blocks[-1][-1]  # periodic BC
            right = u1_blocks[(b+1) % num_blocks][0]

            # Tarefa paralela por bloco
            blk_next = compute_block(u0_blk, u1_blk, left, right, dx, dt, delta)

            next_blocks.append(blk_next)

        # Atualiza blocos
        u0_blocks = u1_blocks
        u1_blocks = next_blocks

        # snapshots periódicos
        if step % skip_steps == 0:
            full = join_blocks(u1_blocks)
            write_snapshot_csv(full, dx, step, outdir)

            p = momentum(full)
            print(f"[t={step*dt:.4f}] momentum={p}")

    # Finalização
    u_final = compss_wait_on(join_blocks(u1_blocks))
    print("Simulação paralela finalizada.")

    return u_final


# Execução direta
if __name__ == "__main__":
    import sys
    N = 512
    t_max = 5.0
    delta = 0.022
    num_blocks = 8

    if len(sys.argv) > 1: N = int(sys.argv[1])
    if len(sys.argv) > 2: t_max = float(sys.argv[2])
    if len(sys.argv) > 3: delta = float(sys.argv[3])
    if len(sys.argv) > 4: num_blocks = int(sys.argv[4])

    KdV_parallel(N, t_max, delta, num_blocks)
