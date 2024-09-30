# utils.py

import torch
import torch.nn as nn
from torchdiffeq import odeint
import os

# Define UNet
class UNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=64):
        super(UNet, self).__init__()
        input_dim = 1 + state_dim  # t and x
        output_dim = state_dim     # u has same dimension as x
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, t, x):
        tx = torch.cat((t, x), dim=1)
        u = self.net(tx)
        # Enforce control constraint ||u|| <= 1
        u_norm = torch.norm(u, dim=1, keepdim=True)
        u_clipped = u / torch.clamp(u_norm, min=1.0)
        return u_clipped

# Define JNet
class JNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=64):
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
        tx = torch.cat((t, x), dim=1)
        return self.net(tx)

# Define dynamics function for ODE integration
def dynamics(t, x, u_net):
    # x: [batch_size, state_dim]
    t_tensor = t * torch.ones(x.shape[0], 1, device=x.device)
    u = u_net(t_tensor, x)
    return u

# Training function for a single stage
def TrainStage(T_start, T_end, u_net, J_net, optimizer, num_epochs=1000, state_dim=2):
    training_losses = []
    for epoch in range(num_epochs):
        # Sample minibatch of initial points at time T_start
        M = 128
        t_j = torch.ones(M, 1) * T_start
        x_j = torch.rand(M, state_dim) * 4 - 2  # x_j in [-2,2]^state_dim
        
        # Sample minibatch of terminal points at time T_end
        M_T = 128
        t_T = torch.ones(M_T, 1) * T_end
        x_T = torch.rand(M_T, state_dim) * 4 - 2  # x_k in [-2,2]^state_dim
        
        # Ensure t_j and x_j require gradients
        t_j = t_j.clone().detach().requires_grad_(True)
        x_j = x_j.clone().detach().requires_grad_(True)
        
        # Compute residuals
        l1, l2, l3 = compute_residuals(t_j, x_j, u_net, J_net)
        
        # Compute terminal loss l4
        J_T = J_net(t_T, x_T)
        h_T = (x_T ** 2).sum(dim=1, keepdim=True)
        l4 = ((J_T - h_T) ** 2).mean()
        
        # Compute cost-to-go losses l5 and l6
        t_span = torch.tensor([T_start, T_end]).to(t_j.device)
        x_t = odeint(lambda t, x: dynamics(t, x, u_net), x_j, t_span)
        x_Tj = x_t[-1]  # x_Tj shape: [M, state_dim]
        h_Tj = (x_Tj ** 2).sum(dim=1, keepdim=True)
        J = J_net(t_j, x_j)
        l5 = (h_Tj ** 2).mean()
        l6 = ((J - h_Tj) ** 2).mean()
        
        # Total loss
        loss = l1 + l2 + l3 + l4 + l5 + l6
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        training_losses.append(loss.item())
        
        # Print losses every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')
            
    return training_losses

# Compute residuals (same as before)
def compute_residuals(t, x, u_net, J_net):
    # (Implementation as in previous code)
    # Ensure t and x require gradients
    t = t.clone().detach().requires_grad_(True)  # [batch_size, 1]
    x = x.clone().detach().requires_grad_(True)  # [batch_size, state_dim]

    # Forward pass
    u = u_net(t, x)  # [batch_size, state_dim]
    J = J_net(t, x)  # [batch_size, 1]

    # Compute gradients
    grad_J = torch.autograd.grad(J.sum(), x, create_graph=True)[0]     # [batch_size, state_dim]
    grad_J_t = torch.autograd.grad(J.sum(), t, create_graph=True)[0]  # [batch_size, 1]

    # Compute grad_J_xt: gradient of grad_J_t w.r.t x
    grad_J_xt = torch.autograd.grad(grad_J_t.sum(), x, create_graph=True)[0]  # [batch_size, state_dim]

    # Compute Hessian-vector product grad_J_xx_u = ∇^2_x J * u
    grad_J_xx_u = torch.autograd.grad(grad_J, x, grad_outputs=u, create_graph=True)[0]  # [batch_size, state_dim]

    # Compute Jacobian-vector product grad_u_T_grad_J = (∇_x u)^T * grad_J
    grad_u_T_grad_J = torch.autograd.grad(u, x, grad_outputs=grad_J, create_graph=True)[0]  # [batch_size, state_dim]

    # Compute l1 residual
    l1_residual = grad_J_xt + grad_J_xx_u + grad_u_T_grad_J  # [batch_size, state_dim]
    l1 = (l1_residual.norm(2, dim=1) ** 2).mean()

    # Compute second derivative of J w.r.t t
    grad_J_tt = torch.autograd.grad(grad_J_t.sum(), t, create_graph=True)[0]  # [batch_size, 1]

    # Compute (∇xt J) ⋅ u
    grad_J_xt_u = (grad_J_xt * u).sum(dim=1, keepdim=True)  # [batch_size, 1]

    # Compute l2 residual
    l2_residual = grad_J_tt + grad_J_xt_u  # [batch_size, 1]
    l2 = (l2_residual ** 2).mean()

    # Compute grad_J_t + grad_J_x^T u (HJB equation)
    grad_J_t_u = grad_J_t + (grad_J * u).sum(dim=1, keepdim=True)  # [batch_size, 1]

    # Compute l3 residual
    l3_residual = grad_J_t_u  # [batch_size, 1]
    l3 = (l3_residual ** 2).mean()

    return l1, l2, l3

# Function to save models
def save_models(u_net, J_net, output_dir, stage):
    u_net_path = os.path.join(output_dir, f'u_net_stage_{stage}.pth')
    J_net_path = os.path.join(output_dir, f'J_net_stage_{stage}.pth')
    torch.save(u_net.state_dict(), u_net_path)
    torch.save(J_net.state_dict(), J_net_path)

# Function to compute ground truth
def compute_ground_truth(t, x, T):
    x_norm = torch.norm(x, dim=1, keepdim=True)  # L2 norm of x(t)
    u_star = torch.zeros_like(x)                 # Initialize u^*(t)
    non_zero_norm_indices = (x_norm > 0).squeeze()
    if non_zero_norm_indices.sum() > 0:
        x_non_zero = x[non_zero_norm_indices]
        x_norm_non_zero = x_norm[non_zero_norm_indices]
        u_star[non_zero_norm_indices, :] = -x_non_zero / x_norm_non_zero

    # Compute the ground truth J^*(t,x)
    J_star = torch.zeros_like(x_norm)
    condition_indices = (x_norm > (T - t))
    J_star[condition_indices] = (x_norm[condition_indices] - (T - t[condition_indices])) ** 2

    return u_star, J_star

# Function to load models
def load_models(model_dir, state_dim):
    u_net = UNet(state_dim=state_dim)
    J_net = JNet(state_dim=state_dim)
    # Load the models from the last stage (stage 0)
    u_net.load_state_dict(torch.load(os.path.join(model_dir, 'u_net_stage_0.pth')))
    J_net.load_state_dict(torch.load(os.path.join(model_dir, 'J_net_stage_0.pth')))
    u_net.eval()
    J_net.eval()
    return u_net, J_net
