# train.py

import torch
import torch.optim as optim
import os
import argparse
from utils import UNet, JNet, TrainStage, save_models

def main():
    parser = argparse.ArgumentParser(description='Train U-net and J-net for Resource Allocation Problem')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs per stage')
    args = parser.parse_args()

    # Problem parameters
    gamma = 0.5
    state_dim = 1  # One-dimensional state variable x(t)
    T = 4.0
    K = 5  # Number of stages
    T_stages = torch.linspace(0, T, steps=K+1)

    output_dir = '../models/resource_allocation'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize networks
    u_net_model = UNet(state_dim=state_dim)
    j_net_model = JNet(state_dim=state_dim)

    # Define optimizer
    optimizer = optim.Adam(list(u_net_model.parameters()) + list(j_net_model.parameters()), lr=1e-3)

    # Create lists to store losses
    training_losses = []

    # Staged training procedure
    for k in range(K-1, -1, -1):
        T_start = T_stages[k].item()
        T_end = T
        print(f'\nTraining on stage {k}, time interval [{T_start}, {T_end}]')
        stage_losses = TrainStage(T_start, T_end, u_net_model, j_net_model, optimizer,
                                  num_epochs=args.num_epochs, state_dim=state_dim, gamma=gamma)
        training_losses.extend(stage_losses)

        # Save model checkpoints
        save_models(u_net_model, j_net_model, output_dir, stage=k)

    # Save training losses for plotting
    os.makedirs('../results/resource_allocation/data', exist_ok=True)
    torch.save(training_losses, f'../results/resource_allocation/data/training_losses.pt')

if __name__ == '__main__':
    main()
