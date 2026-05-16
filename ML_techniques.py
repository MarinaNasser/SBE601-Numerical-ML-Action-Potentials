import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# Import your classical numerical solver module
from finite_difference_solver import run_finite_difference_solver

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Base PINN Model Architecture
# =====================================================================
class ActionPotentialPINN(nn.Module):
    def __init__(self):
        super(ActionPotentialPINN, self).__init__()
        # Deep network architecture to handle sharp non-linear spikes
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

# =====================================================================
# 2. Optimized PDE & Loss Engine
# =====================================================================
def pde_loss(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    u = model(t, x)
    
    # Calculate derivatives via Automatic Differentiation
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    du_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
    
    # Nagumo Constants matching textbook equation (4.1a)
    D = 0.01   
    a = 0.25   
    
    # 1. Interior PDE Residual
    residual_u = du_dt - (D * du_dx2) + (u * (1.0 - u) * (a - u))
    loss_pde = torch.mean(residual_u**2)
    
    # 2. Strict Boundary Conditions (Enforces the persistent electrical stimulus at x=0)
    t_b = torch.rand(100, 1)
    u_b_left = model(t_b, torch.zeros_like(t_b))
    loss_bc = torch.mean((u_b_left - 1.0)**2)
    
    # 3. Initial Conditions (Enforces the system state at t=0)
    x_i = torch.rand(100, 1)
    u_init_pred = model(torch.zeros_like(x_i), x_i)
    # Target initial condition: resting state everywhere except the stimulus zone
    u_init_target = torch.where(x_i < 0.15, torch.ones_like(x_i), torch.zeros_like(x_i))
    loss_ic = torch.mean((u_init_pred - u_init_target)**2)
    
    # Highly weighted combination to prevent the model from choosing flat "lazy" minimums
    return 5.0 * loss_pde + 15.0 * loss_bc + 15.0 * loss_ic
# =====================================================================
# 3. Enhanced Training Schemes
# =====================================================================

# SCHEME 1: Standard PINN
def train_standard_pinn(model, epochs=20000):
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    # Gradually decay learning rate to fine-tune around the sharp wavefront
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        t = torch.rand(300, 1)
        x = torch.rand(300, 1)
        
        loss = pde_loss(model, t, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 5000 == 0:
            print(f"  Epoch {epoch}/{epochs} - Loss: {loss.item():.5f}")
    return model

# SCHEME 2: Data-Informed PINN (Hybrid)
def train_hybrid_pinn(model, data_t, data_x, data_u, epochs=15000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    for epoch in range(epochs):
        loss_pde = pde_loss(model, torch.rand(150, 1), torch.rand(150, 1))
        
        pred = model(data_t, data_x)
        loss_data = torch.mean((pred - data_u)**2)
        
        # Balance out the physics guidance with the ground-truth data points
        total_loss = loss_pde + 30 * loss_data  
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
    return model  

# SCHEME 3: Discrete-Time Neural Step Solver
class NeuralStepSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), 
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, u_n, dt):
        return u_n + dt * self.net(u_n)

def pre_train_step_solver(model, U_numerical, dt, epochs=6000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train across the full sequential grid matrix pairs
    inputs = torch.tensor(U_numerical[:-1, :], dtype=torch.float32).flatten().view(-1, 1)
    targets = torch.tensor(U_numerical[1:, :], dtype=torch.float32).flatten().view(-1, 1)
    
    for epoch in range(epochs):
        pred_next = inputs + dt * model.net(inputs)
        loss = torch.mean((pred_next - targets)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# =====================================================================
# 4. Main Pipeline Execution
# =====================================================================
if __name__ == "__main__":
    # --- STEP 0: EXTRACT DATA FROM SEPARATE MODULE ---
    t_numerical, x_numerical, U_numerical = run_finite_difference_solver()
    
    # Uniformly sample dense coordinates for the Hybrid PINN solver
    num_samples = 500
    sampled_t_idx = np.random.choice(len(t_numerical), num_samples)
    sampled_x_idx = np.random.choice(len(x_numerical), num_samples)

    data_t = torch.tensor(t_numerical[sampled_t_idx], dtype=torch.float32).view(-1, 1)
    data_x = torch.tensor(x_numerical[sampled_x_idx], dtype=torch.float32).view(-1, 1)
    data_u = torch.tensor(U_numerical[sampled_t_idx, sampled_x_idx], dtype=torch.float32).view(-1, 1)

    # Output grid setups (Evaluating space profiles exactly at t = 0.5)
    x_grid = torch.linspace(0, 1, 100).view(-1, 1)
    t_snapshot = torch.ones_like(x_grid) * 0.5

    # --- SCHEME 1 ---
    print("\n--- Training Scheme 1: Standard PINN (Optimized Strategy) ---")
    model_1 = ActionPotentialPINN()
    start_1 = time.time()
    train_standard_pinn(model_1, epochs=20000)
    time_1 = time.time() - start_1

    with torch.no_grad():
        u_pred_1 = model_1(t_snapshot, x_grid).numpy()

    # --- SCHEME 2 ---
    print("\n--- Training Scheme 2: Hybrid PINN (Optimized Weights) ---")
    model_2 = ActionPotentialPINN()
    start_2 = time.time()
    train_hybrid_pinn(model_2, data_t, data_x, data_u, epochs=15000)
    time_2 = time.time() - start_2

    with torch.no_grad():
        u_pred_2 = model_2(t_snapshot, x_grid).numpy()

    # --- SCHEME 3 ---
    print("\n--- Training & Running Scheme 3: Neural Step Solver ---")
    model_3 = NeuralStepSolver()
    dt_step = 0.001 
    
    start_train_3 = time.time()
    model_3 = pre_train_step_solver(model_3, U_numerical, dt=dt_step, epochs=6000)
    time_train_3 = time.time() - start_train_3
    print(f"Scheme 3 Offline Pre-training completed in {time_train_3:.2f} seconds.")
    
    # Execution loop
    u_current = torch.zeros(100, 1)
    u_current[0:15, 0] = 1.0  # Apply boundary trigger condition
    
    start_3 = time.time()
    for step in range(500): 
        with torch.no_grad():
            u_current = model_3(u_current, dt_step)
    time_3 = time.time() - start_3
    print(f"Scheme 3 time-marching loop completed in {time_3:.4f} seconds.")

    # --- SAVE INFRASTRUCTURE OUTPUTS ---
    u_pred_3 = u_current.numpy() 

    # Cleaned line assignment parsing the accurate textbook array indices
    t_index_05 = (np.abs(t_numerical - 0.5)).argmin()
    u_ground_truth = U_numerical[t_index_05, :]

    # =====================================================================
    # 5. THE GRAPHICAL OUTPUT SUITE (Requirements #8, #7 & #12)
    # =====================================================================
    print("\nGenerating final comparative figures...")

    # --- PLOT 1: Comprehensive Line Spatial Profile ---
    plt.figure(figsize=(10, 5.5))
    plt.plot(x_numerical, u_ground_truth, label="Ground Truth (Finite Difference)", color="black", linewidth=2.5)
    plt.plot(x_grid.numpy(), u_pred_1, label="Scheme 1: Standard PINN", color="blue", linestyle="--")
    plt.plot(x_grid.numpy(), u_pred_2, label="Scheme 2: Hybrid PINN", color="green", linestyle="-.")
    plt.plot(x_grid.numpy(), u_pred_3, label="Scheme 3: Neural Step Solver", color="red", linestyle=":")
    
    plt.title("Action Potential Wave Propagation Profile Comparison at Time = 0.5")
    plt.xlabel("Position along Nerve Fiber (x)")
    plt.ylabel("Membrane Potential (u)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.ylim(-0.1, 1.1)
    plt.savefig("plot_1_spatial_profile.png", dpi=300)
    plt.show()

    # --- PLOT 2: Spatiotemporal Continuous 2D Heatmaps ---
    t_mesh = torch.linspace(0, 1, 100).view(-1, 1)
    T, X = torch.meshgrid(t_mesh.squeeze(), x_grid.squeeze(), indexing='ij')
    mesh_input = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        V_mesh_1 = model_1(mesh_input[:, 0:1], mesh_input[:, 1:2]).reshape(100, 100).numpy()
        V_mesh_2 = model_2(mesh_input[:, 0:1], mesh_input[:, 1:2]).reshape(100, 100).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ground Truth Heatmap
    im0 = axes[0].imshow(U_numerical, extent=[0, 1, 0, 1], origin='lower', cmap='jet', aspect='auto')
    axes[0].set_title("Ground Truth (Finite Difference) Field")
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel("Space (x)")
    fig.colorbar(im0, ax=axes[0])

    # Scheme 1 Heatmap
    im1 = axes[1].contourf(T.numpy(), X.numpy(), V_mesh_1, cmap='jet', levels=50)
    axes[1].set_title("Standard PINN Continuous Field")
    axes[1].set_xlabel("Time (t)")
    fig.colorbar(im1, ax=axes[1])

    # Scheme 2 Heatmap
    im2 = axes[2].contourf(T.numpy(), X.numpy(), V_mesh_2, cmap='jet', levels=50)
    axes[2].set_title("Hybrid PINN Field")
    axes[2].set_xlabel("Time (t)")
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("plot_2_heatmap_comparison.png", dpi=300)
    plt.show()

    # --- PLOT 3: Execution Metric Profiling Bar Chart ---
    plt.figure(figsize=(8, 5))
    schemes = ['Standard PINN', 'Hybrid PINN', 'Neural Step Solver\n(Inference Loop)']
    execution_times = [time_1, time_2, (time_train_3 + time_3)]
    bars = plt.bar(schemes, execution_times, color=['blue', 'green', 'red'], edgecolor='black', alpha=0.8)
    plt.ylabel('Execution/Training Time Window (Seconds)')
    plt.title('SBE601 System Computational Efficiency Profile')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(execution_times)*0.01),
                 f'{height:.4f}s', ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig("plot_3_performance_metrics.png", dpi=300)
    plt.show()
    print("Process Finished. Review your updated workspace images.")