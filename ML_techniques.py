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





# Initialize model
pinn_model = ActionPotentialPINN()

# --- RUNNING SCHEME 1: Standard PINN ---
start_time = time.time()
print("Starting Standard PINN Training...")
trained_pinn = train_standard_pinn(pinn_model, epochs=10000)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.") # For Requirement #7

# --- GENERATING RESULTS ---
# Create a grid for testing (Time vs. Space)
t_test = torch.linspace(0, 1, 100).view(-1, 1)
x_test = torch.linspace(0, 1, 100).view(-1, 1)

# Correctly create the 2D grid
T, X = torch.meshgrid(t_test.squeeze(), x_test.squeeze(), indexing='ij')

# Reshape into (N, 2) where N = 10000
# We need only TWO columns: one for T and one for X
test_input = torch.stack([T.flatten(), X.flatten()], dim=1)

# Predict with AI
with torch.no_grad():
    predictions = trained_pinn.net(test_input) # Directly call the network
    # V is the first output column (membrane potential)
    V_pred = predictions[:, 0].reshape(100, 100).numpy()

# --- PLOTTING (For Requirement #8 & #12) ---
plt.figure(figsize=(10, 4))
plt.contourf(T.numpy(), X.numpy(), V_pred, cmap='jet')
plt.colorbar(label='Membrane Potential (V)')
plt.title("AI-Generated Action Potential Propagation")
plt.xlabel("Time (t)")
plt.ylabel("Space (x)")
plt.show() # This plot goes in your 4-page report and presentation