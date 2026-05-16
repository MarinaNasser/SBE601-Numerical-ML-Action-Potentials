import numpy as np
import torch

def run_finite_difference_solver():
    print("--- Running Classical Numerical Solver (Ground Truth) ---")
    # Grid parameters
    N_x = 100  # Spatial points
    N_t = 1000 # Time steps (high density for numerical stability)
    
    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 1, N_t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Equation Constants from textbook
    D = 0.01
    a = 0.25
    
    # Check Stability Criterion for Explicit PDE Solvers: dt <= (dx^2) / (2*D)
    stability_limit = (dx**2) / (2 * D)
    if dt > stability_limit:
        print(f"Warning: numerical scheme may diverge. dt ({dt}) > limit ({stability_limit})")

    # Initialize solution matrix: Rows = Time, Columns = Space
    U = np.zeros((N_t, N_x))
    
    # Apply Initial Condition: Initial biological boundary stimulus
    # A localized voltage spike at the beginning of the nerve fiber
    U[0, 0:15] = 1.0 
    
    # Time-marching loop (Finite Difference)
    for n in range(0, N_t - 1):
        for i in range(1, N_x - 1):
            # 2nd derivative approximation (Central Difference)
            d2u_dx2 = (U[n, i+1] - 2*U[n, i] + U[n, i-1]) / (dx**2)
            
            # Cubic reaction term matching textbook Equation (4.1a)
            reaction = - U[n, i] * (1.0 - U[n, i]) * (a - U[n, i])
            
            # Update next time step
            U[n+1, i] = U[n, i] + dt * (D * d2u_dx2 - reaction)
            
        # Neumann Boundary Conditions (Insulated ends: du/dx = 0)
        U[n+1, 0] = U[n+1, 1]
        U[n+1, -1] = U[n+1, -2]
        
    print("Numerical solution matrix completed successfully.")
    return t, x, U