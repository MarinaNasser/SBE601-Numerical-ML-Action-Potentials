import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. Base PINN Model Architecture
class ActionPotentialPINN(nn.Module):
    def __init__(self):
        super(ActionPotentialPINN, self).__init__()
        # Input: (t, x) | Output: (v, w)
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2) 
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

# 2. PDE Residual Calculation (The "Physics" Part)
def pde_loss(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    pred = model(t, x)
    v, w = pred[:, 0:1], pred[:, 1:2]
    
    # Automatic Differentiation
    dv_dt = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    dv_dx = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    dv_dx2 = torch.autograd.grad(dv_dx, x, torch.ones_like(dv_dx), create_graph=True)[0]
    dw_dt = torch.autograd.grad(w, t, torch.ones_like(w), create_graph=True)[0]
    
    # FN Parameters
    D, a, b, eps = 0.01, 0.7, 0.8, 0.08
    
    # Residuals
    res_v = dv_dt - D*dv_dx2 - (v - (v**3)/3 - w)
    res_w = dw_dt - eps*(v + a - b*w)
    
    return torch.mean(res_v**2) + torch.mean(res_w**2)

# --- THE THREE SCHEMES ---

# SCHEME 1: Standard PINN (Vanilla Physics-Informed)
# Focuses on solving the forward problem using collocation points.
def train_standard_pinn(model, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        # Generate random interior points
        t = torch.rand(100, 1)
        x = torch.rand(100, 1)
        
        loss = pde_loss(model, t, x) # Pure physics loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# SCHEME 2: Data-Informed PINN (Hybrid)
# Uses a small set of "experimental" or "numerical" data to guide the spike.
def train_hybrid_pinn(model, data_t, data_x, data_v, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        # Physics Loss + Data Loss
        loss_pde = pde_loss(model, torch.rand(50, 1), torch.rand(50, 1))
        
        pred = model(data_t, data_x)
        loss_data = torch.mean((pred[:, 0:1] - data_v)**2)
        
        total_loss = loss_pde + 10 * loss_data # Weighted data loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# SCHEME 3: Discrete-Time Neural PDE (Runge-Kutta Informed)
# Instead of continuous (t, x), it predicts the next time step (t+dt)
# essentially mimicking a numerical solver.
class NeuralStepSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
        
    def forward(self, v_n, w_n, dt):
        # Learns the residual 'k' such that u_{n+1} = u_n + dt * k
        res = self.net(torch.cat([v_n, w_n], dim=1))
        return v_n + dt * res[:, 0:1], w_n + dt * res[:, 1:2]


#########################################################################################

# Setup the testing grid (100x100 space-time domain)
t_test = torch.linspace(0, 1, 100).view(-1, 1)
x_test = torch.linspace(0, 1, 100).view(-1, 1)
T, X = torch.meshgrid(t_test.squeeze(), x_test.squeeze(), indexing='ij')
test_input = torch.stack([T.flatten(), X.flatten()], dim=1)

# ==========================================
# 1. RUN & TEST SCHEME 1: Standard PINN
# ==========================================
print("--- Training Scheme 1: Standard PINN ---")
model_1 = ActionPotentialPINN()
start_1 = time.time()
train_standard_pinn(model_1, epochs=10000)
time_1 = time.time() - start_1
print(f"Scheme 1 completed in {time_1:.2f} seconds.\n")

with torch.no_grad():
    pred_1 = model_1.net(test_input)
    V_pred_1 = pred_1[:, 0].reshape(100, 100).numpy()


# ==========================================
# 2. RUN & TEST SCHEME 2: Hybrid PINN
# ==========================================
print("--- Training Scheme 2: Hybrid PINN ---")
model_2 = ActionPotentialPINN()

# MOCK DATA: Replace these placeholders with actual array selections 
# extracted from your traditional numerical solver grid!
mock_t = torch.rand(50, 1)
mock_x = torch.rand(50, 1)
mock_v = torch.sin(mock_t) * torch.cos(mock_x) # Your numerical solver outputs go here

start_2 = time.time()
train_hybrid_pinn(model_2, mock_t, mock_x, mock_v, epochs=10000)
time_2 = time.time() - start_2
print(f"Scheme 2 completed in {time_2:.2f} seconds.\n")

with torch.no_grad():
    pred_2 = model_2.net(test_input)
    V_pred_2 = pred_2[:, 0].reshape(100, 100).numpy()


# ==========================================
# 3. RUN & TEST SCHEME 3: Neural Step Solver
# ==========================================
print("--- Running Scheme 3: Neural Step Solver ---")
model_3 = NeuralStepSolver()
dt = 0.01

# Initialize starting boundary states (e.g., resting potential over space)
v_initial = torch.zeros(100, 1) 
w_initial = torch.zeros(100, 1)

# Add an initial stimulus trigger at the boundary to initiate a propagation spike
v_initial[0:10, 0] = 1.0 

v_current = v_initial.clone()
w_current = w_initial.clone()
v_history = [v_current.numpy().flatten()]

start_3 = time.time()
# March sequentially step-by-step through the 100 time increments
for step in range(99):
    with torch.no_grad():
        v_next, w_next = model_3(v_current, w_current, dt)
        v_history.append(v_next.numpy().flatten())
        v_current, w_current = v_next, w_next
time_3 = time.time() - start_3
print(f"Scheme 3 time-marching completed in {time_3:.2f} seconds.\n")

V_pred_3 = np.array(v_history) # Formulates a 100x100 space-time array


# ==========================================
# VISUAL COMPARISON (Requirement #8 & #12)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Scheme 1
im1 = axes[0].contourf(T.numpy(), X.numpy(), V_pred_1, cmap='jet')
axes[0].set_title(f"Standard PINN\nTime: {time_1:.1f}s")
axes[0].set_xlabel("Time (t)")
axes[0].set_ylabel("Space (x)")
fig.colorbar(im1, ax=axes[0])

# Plot Scheme 2
im2 = axes[1].contourf(T.numpy(), X.numpy(), V_pred_2, cmap='jet')
axes[1].set_title(f"Hybrid PINN\nTime: {time_2:.1f}s")
axes[1].set_xlabel("Time (t)")
fig.colorbar(im2, ax=axes[1])

# Plot Scheme 3
im3 = axes[2].contourf(T.numpy(), X.numpy(), V_pred_3, cmap='jet')
axes[2].set_title(f"Neural Step Solver\nTime: {time_3:.1f}s")
axes[2].set_xlabel("Time (t)")
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("pinn_approaches_comparison.png")
plt.show()