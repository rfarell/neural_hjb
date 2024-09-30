# utils.py

import torch
import torch.nn as nn
from torchdiffeq import odeint
import os

# Define UNet
class UNet(nn.Module):
    def __init__(self, state_dim=1, hidden_dim=64):
        super(UNet, self).__init__()
        input_dim = 1 + state_dim  # t and x
        output_dim = 1             # u is scalar in [0,1]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensures output is in (0,1)
        )
        
    def forward(self, t, x):
        # Ensure t and x are 2D tensors
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # Shape: [batch_size, 1]
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(1)  # Shape: [1, state_dim]
        elif x.dim() == 1:
            x = x.unsqueeze(1)  # Shape: [batch_size, state_dim]
        tx = torch.cat((t, x), dim=1)
        u = self.net(tx)
        return u

# Define JNet
class JNet(nn.Module):
    def __init__(self, state_dim=1, hidden_dim=64):
        super(JNet, self).__init__()
        input_dim = 1 + state_dim  # t and x
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t, x):
        # Ensure t and x are 2D tensors
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1]
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # Shape: [batch_size, 1]
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(1)  # Shape: [1, state_dim]
        elif x.dim() == 1:
            x = x.unsqueeze(1)  # Shape: [batch_size, state_dim]
        tx = torch.cat((t, x), dim=1)
        return self.net(tx)

# Define augmented dynamics function for ODE integration
def augmented_dynamics(t, y, u_net, gamma):
    # y: [batch_size, 2], where y[:, 0] = x, y[:, 1] = c
    if y.dim() == 1:
        y = y.unsqueeze(0)  # Shape: [1, 2]
    x = y[:, 0:1]  # Shape: [batch_size, 1]
    c = y[:, 1:2]  # Shape: [batch_size, 1] (not used in dynamics)

    # Prepare t_tensor to match the batch size of x
    t_tensor = t * torch.ones_like(x)  # Shape: [batch_size, 1]

    # Compute control u(t, x)
    u = u_net(t_tensor, x)  # Shape: [batch_size, 1]

    # Compute dynamics
    dxdt = gamma * u * x  # Shape: [batch_size, 1]
    dcdt = (1 - u) * x    # Shape: [batch_size, 1]

    dy_dt = torch.cat([dxdt, dcdt], dim=1)  # Shape: [batch_size, 2]

    return dy_dt.squeeze(0) if dy_dt.shape[0] == 1 else dy_dt

# Training function for a single stage
def TrainStage(T_start, T_end, u_net, j_net, optimizer, num_epochs=1000, state_dim=1, gamma=0.5):
    training_losses = []
    for epoch in range(num_epochs):
        # Sample minibatch of initial points at time T_start
        M = 128
        t_j = torch.ones(M, 1) * T_start
        x_j = torch.rand(M, state_dim) * 4 + 1  # x_j in [1,5]

        # Sample minibatch of terminal points at time T_end
        M_T = 128
        t_T = torch.ones(M_T, 1) * T_end
        x_T = torch.rand(M_T, state_dim) * 4 + 1  # x_k in [1,5]

        # Ensure t_j and x_j require gradients
        t_j = t_j.clone().detach().requires_grad_(True)
        x_j = x_j.clone().detach().requires_grad_(True)

        # Compute residuals
        l1, l2, l3 = compute_residuals(t_j, x_j, u_net, j_net, gamma)

        # Terminal condition (no terminal cost)
        J_T = j_net(t_T, x_T)
        h_T = torch.zeros_like(J_T)
        l4 = ((J_T - h_T) ** 2).mean()

        # Simulate augmented trajectory from t_j to T_end
        t_span = torch.tensor([T_start, T_end]).to(t_j.device)
        y0 = torch.cat([x_j, torch.zeros_like(x_j)], dim=1)  # Initial augmented state [x_j, c=0]
        y_t = odeint(lambda t, y: augmented_dynamics(t, y, u_net, gamma), y0, t_span)
        y_Tj = y_t[-1]  # Shape: [batch_size, 2]
        x_Tj = y_Tj[:, 0:1]  # Final state x(T)
        C_j = y_Tj[:, 1:2]   # Cumulative cost from t_j to T_end

        # Compute l5 and l6
        l5 = (C_j ** 2).mean()
        J = j_net(t_j, x_j)
        l6 = ((J - C_j) ** 2).mean()

        # Total loss
        loss = l1 + l2 + l3 + l4 - l5 + l6

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        training_losses.append(loss.item())

        # Print losses every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    return training_losses

# Compute residuals
def compute_residuals(t, x, u_net, j_net, gamma):
    # Ensure t and x require gradients
    t = t.clone().detach().requires_grad_(True)  # [batch_size, 1]
    x = x.clone().detach().requires_grad_(True)  # [batch_size, 1]

    # Forward pass
    u = u_net(t, x)          # [batch_size, 1]
    J = j_net(t, x)          # [batch_size, 1]

    # Compute dynamics
    f = gamma * u * x        # [batch_size, 1]

    # Compute running cost (Note: we maximize (1 - u) x, so g = - (1 - u) x)
    g = - (1 - u) * x        # [batch_size, 1]

    # Compute gradients
    grad_J = torch.autograd.grad(J.sum(), x, create_graph=True)[0]     # [batch_size, 1]
    grad_J_t = torch.autograd.grad(J.sum(), t, create_graph=True)[0]   # [batch_size, 1]

    # Compute l1 residual: HJB equation residual
    HJB_residual = g + grad_J_t + grad_J * f
    l1 = (HJB_residual ** 2).mean()

    # Compute l2 residual: Gradient w.r.t x
    # Since the state is 1D, grad_J_xx is scalar
    grad_J_xx = torch.autograd.grad(grad_J.sum(), x, create_graph=True)[0]  # [batch_size, 1]
    l2_residual = grad_J_xx + grad_J * gamma * u
    l2 = (l2_residual ** 2).mean()

    # Compute l3 residual: Second derivative w.r.t t
    grad_J_tt = torch.autograd.grad(grad_J_t.sum(), t, create_graph=True)[0]  # [batch_size, 1]
    l3_residual = grad_J_tt + grad_J_t * f
    l3 = (l3_residual ** 2).mean()

    return l1, l2, l3

# Function to save models
def save_models(u_net, j_net, output_dir, stage):
    u_net_path = os.path.join(output_dir, f'u_net_stage_{stage}.pth')
    j_net_path = os.path.join(output_dir, f'j_net_stage_{stage}.pth')
    torch.save(u_net.state_dict(), u_net_path)
    torch.save(j_net.state_dict(), j_net_path)

# Function to compute the optimal control u*(t) using PyTorch
def compute_ground_truth(t, T, gamma):
    """
    Computes the optimal control u*(t) based on the closed-form expression using PyTorch.
    
    Parameters:
    t (torch.Tensor): Time tensor or a single time value.
    T (float): Terminal time.
    gamma (float): Parameter in the adjoint equation.
    
    Returns:
    torch.Tensor: Optimal control value(s) at time t.
    """
    # Compute the switching time
    switching_time = T - 1 / gamma
    
    # Ensure t is a torch tensor
    t = torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t
    
    # Compute u_star using PyTorch tensor operations
    u_star = torch.where(t < switching_time, torch.tensor(1.0), torch.tensor(0.0))
    
    return u_star

# Function to load models
def load_models(model_dir, state_dim):
    u_net = UNet(state_dim=state_dim)
    j_net = JNet(state_dim=state_dim)
    # Load the models from the last stage (stage 0)
    u_net.load_state_dict(torch.load(os.path.join(model_dir, 'u_net_stage_0.pth')))
    j_net.load_state_dict(torch.load(os.path.join(model_dir, 'j_net_stage_0.pth')))
    u_net.eval()
    j_net.eval()
    return u_net, j_net
