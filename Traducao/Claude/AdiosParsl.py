"""
Distributed under the OSI-approved Apache License, Version 2.0.

Solves the initial value problem for the Korteweg de-Vries equation via the
Zabusky and Kruskal scheme using Parsl for workflow management.

See: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.15.240
Zabusky, Norman J., and Martin D. Kruskal. "Interaction of solitons in a 
collisionless plasma and the recurrence of initial states." 
Physical review letters 15.6 (1965): 240.
"""

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import numpy as np
import pickle
import os
import sys


def display_progress(progress):
    """Display a progress bar"""
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
    print(f"] {int(progress * 100.0)}%\033[0m\r", end="")
    sys.stdout.flush()


def momentum(u):
    """
    This momentum is a conserved quantity of the numerical method.
    Use it to sanity check the solution.
    """
    return np.sum(u)


@python_app
def initialize_solution(N, dx):
    """Initialize the solution u(x,0) = cos(πx)"""
    import numpy as np
    u0 = np.cos(np.pi * np.arange(N) * dx)
    return u0


@python_app
def compute_first_step(u0, N, dx, dt):
    """Compute the first time step using analytical approximation"""
    import numpy as np
    u1 = np.zeros(N)
    for i in range(N):
        cdt = np.cos(np.pi * i * dx) * dt
        u1[i] = np.cos(np.pi * (i * dx - cdt))
    return u1


@python_app
def kdv_step(u0, u1, N, dx, dt, delta):
    """
    Perform one time step of the KdV equation using the 
    Zabusky-Kruskal scheme
    """
    import numpy as np
    
    k1 = dt / (3 * dx)
    k2 = delta * delta * dt / (dx * dx * dx)
    u2 = np.zeros(N)
    
    # Boundary point i=0
    t1 = (u1[1] + u1[0] + u1[N-1]) * (u1[1] - u1[N-1])
    t2 = u1[2] - 2*u1[1] + 2*u1[N-1] - u1[N-2]
    u2[0] = u0[0] - k1*t1 - k2*t2
    
    # Boundary point i=1
    t1 = (u1[2] + u1[1] + u1[0]) * (u1[2] - u1[0])
    t2 = u1[3] - 2*u1[2] + 2*u1[0] - u1[N-1]
    u2[1] = u0[1] - k1*t1 - k2*t2
    
    # Interior points
    for i in range(2, N-2):
        t1 = (u1[i+1] + u1[i] + u1[i-1]) * (u1[i+1] - u1[i-1])
        t2 = u1[i+2] - 2*u1[i+1] + 2*u1[i-1] - u1[i-2]
        u2[i] = u0[i] - k1*t1 - k2*t2
    
    # Boundary point i=N-2
    t1 = (u1[N-1] + u1[N-2] + u1[N-3]) * (u1[N-1] - u1[N-3])
    t2 = u1[0] - 2*u1[N-1] + 2*u1[N-3] - u1[N-4]
    u2[N-2] = u0[N-2] - k1*t1 - k2*t2
    
    # Boundary point i=N-1
    t1 = (u1[0] + u1[N-1] + u1[N-2]) * (u1[0] - u1[N-2])
    t2 = u1[1] - 2*u1[0] + 2*u1[N-2] - u1[N-3]
    u2[N-1] = u0[N-1] - k1*t1 - k2*t2
    
    return u2


@python_app
def save_snapshot(u, step, output_dir):
    """Save a snapshot of the solution to disk"""
    import pickle
    import os
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"snapshot_{step:06d}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(u, f)
    return filename


def KdV_parsl(N, dt, t_max, delta=0.022, output_dir="kdv_output"):
    """
    Solve the KdV equation using Parsl workflow
    
    Parameters:
    -----------
    N : int
        Number of spatial grid points
    dt : float
        Time step
    t_max : float
        Maximum simulation time
    delta : float
        Interaction parameter
    output_dir : str
        Directory to save output snapshots
    """
    
    if N <= 0:
        raise ValueError("N > 0 is required")
    if dt > 1:
        raise ValueError("time step is too big")
    if dt <= 0:
        raise ValueError("dt > 0 is required")
    
    dx = 2.0 / N
    M = int(np.ceil(t_max / dt))
    
    print(f"Solving the initial value problem for the KdV equation "
          f"∂tu + u∂ₓu + δ²∂ₓ³u = 0 using δ = {delta}.")
    print(f"Initial conditions: u(x,0) = cos(πx) for x∈[0,2].")
    print(f"Using Δx = {dx}, Δt = {dt} and t_max = {t_max}")
    
    # Initialize solution
    u0_future = initialize_solution(N, dx)
    u0 = u0_future.result()
    
    # Save initial condition
    save_snapshot(u0, 0, output_dir)
    
    # Compute first step
    u1_future = compute_first_step(u0, N, dx, dt)
    u1 = u1_future.result()
    
    # Time stepping loop
    skip_steps = 40000
    saved_steps = [0]
    
    for j in range(1, M-1):
        # Compute next step
        u2_future = kdv_step(u0, u1, N, dx, dt, delta)
        u2 = u2_future.result()
        
        # Check momentum (conserved quantity)
        p = momentum(u2)
        
        if np.isnan(p):
            print(f"\nSolution diverged at t = {(j+1)*dt}")
            print(f"Momentum = {p}")
            return saved_steps
        
        if np.abs(p) > np.sqrt(np.finfo(float).eps):
            print(f"\nWarning: Momentum deviation at t = {(j+1)*dt}")
            print(f"Momentum = {p}")
        
        # Save snapshot periodically
        if (j + 1) % skip_steps == 0:
            display_progress((j + 1) / (M - 1))
            save_snapshot(u2, j+1, output_dir)
            saved_steps.append(j+1)
        
        # Update for next iteration
        u0 = u1
        u1 = u2
    
    print("\nSimulation completed successfully!")
    
    # Save metadata
    metadata = {
        'N': N,
        'dx': dx,
        'dt': dt,
        't_max': t_max,
        'delta': delta,
        'M': M,
        'saved_steps': saved_steps
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    return saved_steps


def setup_parsl(max_threads=4):
    """
    Configure Parsl with ThreadPoolExecutor
    
    Parameters:
    -----------
    max_threads : int
        Maximum number of threads to use (default: 4)
    """
    config = Config(
        executors=[
            ThreadPoolExecutor(
                label="local_threads",
                max_threads=max_threads
            )
        ],
        strategy='simple'
    )
    parsl.load(config)


def main():
    """Main function"""
    N = 256
    t_max = 5.0
    delta = 0.022
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python kdv_parsl.py N t_max δ")
            print("where N is number of spatial gridpoints,")
            print("t_max is max simulation time, and δ is interaction parameter")
            print("e.g., python kdv_parsl.py 512 10 0.022")
            return 0
        N = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        t_max = float(sys.argv[2])
    
    if len(sys.argv) > 3:
        delta = float(sys.argv[3])
    
    dx = 1.0 / N
    dt = 27 * dx * dx * dx / 4
    
    # Setup Parsl
    setup_parsl()
    
    try:
        saved_steps = KdV_parsl(N, dt, t_max, delta)
        print(f"\nSaved {len(saved_steps)} snapshots to kdv_output/")
    except Exception as e:
        print(f"Caught exception from KdV call: {e}")
        return 1
    finally:
        parsl.clear()
    
    return 0


if __name__ == "__main__":
    exit(main())