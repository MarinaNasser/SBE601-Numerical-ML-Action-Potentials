import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# =====================================================================
# 1. Base PINN Model Architecture
# =====================================================================
class ActionPotentialPINN(nn.Module):
    def __init__(self):
        super(ActionPotentialPINN, self).__init__()
        # Input: (t, x) | Output: u (Membrane Potential)
        # Updated to output 1 value to perfectly match the 1-variable Nagumo PDE
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1) 
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

# =====================================================================
# 2. PDE Residual Calculation (The "Physics" Part)
# =====================================================================
def pde_loss(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    # Predict u (membrane potential) using the model
    u = model(t, x)
    
    # Calculate derivatives using automatic differentiation
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    du_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
    
    # Parameters from the textbook chapter
    D = 0.01   # Diffusion coefficient
    a = 0.25   # Threshold parameter
    
    # --- Exact Match to Nagumo Equation (4.1a) ---
    # du/dt = D * d^2u/dx^2 - u*(1-u)*(a-u)
    residual_u = du_dt - (D * du_dx2) + (u * (1.0 - u) * (a - u))
    
    return torch.mean(residual_u**2)

# =====================================================================
# 3. The Three Deep Learning Training Schemes
# =====================================================================

# SCHEME 1: Standard PINN (Vanilla Physics-Informed)
def train_standard_pinn(model, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        t = torch.rand(100, 1)
        x = torch.rand(100, 1)
        
        loss = pde_loss(model, t, x)  # Pure physics loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# SCHEME 2: Data-Informed PINN (Hybrid)
def train_hybrid_pinn(model, data_t, data_x, data_u, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        # Physics Loss
        loss_pde = pde_loss(model, torch.rand(50, 1), torch.rand(50, 1))
        
        # Labeled Data Loss (The Cheat Sheet)
        pred = model(data_t, data_x)
        loss_data = torch.mean((pred - data_u)**2)
        
        total_loss = loss_pde + 10 * loss_data  # Weighted combination
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return model  # Fixed: Added return statement

# SCHEME 3: Discrete-Time Neural PDE (Neural Step Solver)
class NeuralStepSolver(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: current voltage over space grid | Output: rate of change
        self.net = nn.Sequential(
            nn.Linear(1, 32), 
            nn.Tanh(), 
            nn.Linear(32, 1)
        )
        
    def forward(self, u_n, dt):
        # Predicts how much u changes over a small time step dt
        rate_of_change = self.net(u_n)
        return u_n + dt * rate_of_change

# =====================================================================
# 4. Main Execution and Testing Pipeline
# =====================================================================
if __name__ == "__main__":
    # Setup spatial positions for evaluation (e.g., 100 points along a nerve fiber)
    x_grid = torch.linspace(0, 1, 100).view(-1, 1)

    # --- EXECUTE SCHEME 1: Standard PINN ---
    print("--- Training Scheme 1: Standard PINN ---")
    model_1 = ActionPotentialPINN()
    start_1 = time.time()
    train_standard_pinn(model_1, epochs=3000)
    time_1 = time.time() - start_1
    print(f"Scheme 1 completed in {time_1:.2f} seconds.\n")

    # Evaluate Scheme 1 at Time = 0.5
    t_snapshot = torch.ones_like(x_grid) * 0.5
    with torch.no_grad():
        u_pred_1 = model_1(t_snapshot, x_grid).numpy()

    # --- EXECUTE SCHEME 2: Hybrid PINN ---
    print("--- Training Scheme 2: Hybrid PINN ---")
    model_2 = ActionPotentialPINN()

    # Generating Mock Numerical Data (Simulating values from your numerical solver code)
    mock_t = torch.rand(100, 1)
    mock_x = torch.rand(100, 1)
    mock_u = 0.5 * (1.0 + torch.sin(mock_x * np.pi)) # Sample wave pattern

    start_2 = time.time()
    train_hybrid_pinn(model_2, mock_t, mock_x, mock_u, epochs=3000)
    time_2 = time.time() - start_2
    print(f"Scheme 2 completed in {time_2:.2f} seconds.\n")

    # Evaluate Scheme 2 at Time = 0.5
    with torch.no_grad():
        u_pred_2 = model_2(t_snapshot, x_grid).numpy()

    # --- EXECUTE SCHEME 3: Neural Step Solver ---
    print("--- Running Scheme 3: Neural Step Solver ---")
    model_3 = NeuralStepSolver()
    dt = 0.01
    
    # Set an initial biological resting state with an explicit starting stimulus trigger
    u_current = torch.zeros(100, 1)
    u_current[0:15, 0] = 1.0  # Stimulating the first 15 segments of the nerve fiber
    
    start_3 = time.time()
    # Progress step-by-step up to time = 0.5 (which equals 50 steps of 0.01)
    for step in range(50):
        with torch.no_grad():
            u_next = model_3(u_current, dt)
            u_current = u_next
    time_3 = time.time() - start_3
    print(f"Scheme 3 time-marching completed in {time_3:.2f} seconds.\n")
    u_pred_3 = u_current.numpy()

    # =====================================================================
    # 5. NEW VISUALIZATION: Intuitive Line Plots (Requirement #8 & #12)
    # =====================================================================
    plt.figure(figsize=(12, 5))

    # Plot lines showing the voltage profile across the nerve at Time = 0.5
    plt.plot(x_grid.numpy(), u_pred_1, label=f"Scheme 1: Standard PINN ({time_1:.1f}s)", color="blue", linestyle="--")
    plt.plot(x_grid.numpy(), u_pred_2, label=f"Scheme 2: Hybrid PINN ({time_2:.1f}s)", color="green", linestyle="-.")
    plt.plot(x_grid.numpy(), u_pred_3, label=f"Scheme 3: Neural Step Solver ({time_3:.1f}s)", color="red", linestyle="-")

    plt.title("Comparison of Action Potential Voltage (u) across Space at Time = 0.5")
    plt.xlabel("Position along Nerve Fiber (x)")
    plt.ylabel("Membrane Potential Voltage (u)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.ylim(-0.2, 1.2)

    # Save and display
    plt.savefig("action_potential_line_comparison.png", dpi=300)
    plt.show()
    print("Generating comprehensive plots... Close each window to proceed.")
    
    # --- PLOT 1: The Intuitive 1D Line Plot (Voltage across Space) ---
    plt.figure(figsize=(10, 5))
    plt.plot(x_grid.numpy(), u_pred_1, label=f"Scheme 1: Standard PINN ({time_1:.2f}s)", color="blue", linestyle="--")
    plt.plot(x_grid.numpy(), u_pred_2, label=f"Scheme 2: Hybrid PINN ({time_2:.2f}s)", color="green", linestyle="-.")
    plt.plot(x_grid.numpy(), u_pred_3, label=f"Scheme 3: Neural Step Solver ({time_3:.2f}s)", color="red", linestyle="-")
    plt.title("Action Potential Voltage (u) Profile Across Space at Time = 0.5")
    plt.xlabel("Position along Nerve Fiber (x)")
    plt.ylabel("Membrane Potential (u)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.ylim(-0.2, 1.2)
    plt.savefig("plot_1_spatial_profile.png", dpi=300)
    plt.show()  # Script pauses here until you close this window

    # Recreate the 2D evaluation grid for the Heatmaps
    t_mesh = torch.linspace(0, 1, 100).view(-1, 1)
    T, X = torch.meshgrid(t_mesh.squeeze(), x_grid.squeeze(), indexing='ij')
    mesh_input = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        V_mesh_1 = model_1(mesh_input[:, 0:1], mesh_input[:, 1:2]).reshape(100, 100).numpy()
        V_mesh_2 = model_2(mesh_input[:, 0:1], mesh_input[:, 1:2]).reshape(100, 100).numpy()

    # --- PLOT 2: Side-by-Side 2D Spatiotemporal Heatmaps (Schemes 1 & 2) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].contourf(T.numpy(), X.numpy(), V_mesh_1, cmap='jet', levels=50)
    axes[0].set_title(f"Standard PINN Propagation Wave\nTraining: {time_1:.1f}s")
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel("Space (x)")
    fig.colorbar(im1, ax=axes[0], label="Potential (u)")

    im2 = axes[1].contourf(T.numpy(), X.numpy(), V_mesh_2, cmap='jet', levels=50)
    axes[1].set_title(f"Hybrid PINN Propagation Wave\nTraining: {time_2:.1f}s")
    axes[1].set_xlabel("Time (t)")
    axes[1].set_ylabel("Space (x)")
    fig.colorbar(im2, ax=axes[1], label="Potential (u)")

    plt.tight_layout()
    plt.savefig("plot_2_heatmap_comparison.png", dpi=300)
    plt.show()  # Script pauses here until you close this window

    # --- PLOT 3: Performance Analysis Bar Chart (Time Metrics) ---
    plt.figure(figsize=(8, 5))
    schemes = ['Standard PINN', 'Hybrid PINN', 'Neural Step Solver']
    execution_times = [time_1, time_2, time_3]
    
    bars = plt.bar(schemes, execution_times, color=['blue', 'green', 'red'], alpha=0.85, edgecolor='black')
    plt.ylabel('Execution Time (seconds)')
    plt.title('SBE601 Computational Efficiency Profile')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(execution_times)*0.01),
                 f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.savefig("plot_3_performance_metrics.png", dpi=300)
    plt.show()  # Final plot window
    print("All plots generated and saved successfully. Script complete.")