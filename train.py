import torch
from n_body_system.dataset_nbody import NBodyDataset
from base_model_extended import EGNN  
import os
from torch import nn, optim
import json
import time

EXP_NAME = "exp_5"
BATCH_SIZE = 100
EPOCHS = 10000
DATASET = "nbody_small"
MAX_TRAINING_SAMPLES = 3000
LR = 3e-4
WEIGHT_DECAY = 1e-12
TEST_INTERVAL = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_mse = nn.MSELoss()

os.makedirs("results", exist_ok=True)
os.makedirs(os.path.join("results", EXP_NAME), exist_ok=True)


def main():
    dataset_train = NBodyDataset(partition='train', dataset_name=DATASET, max_samples=MAX_TRAINING_SAMPLES)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

    model = EGNN(input_dim=1, feature_dim=64, hidden_dim=64, vector_dim=3, n_layers=4, edge_attribnute_dim=2).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    results = {'epochs': [], 'losses': []}
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        train(model, optimizer, epoch, loader_train)
        if epoch % TEST_INTERVAL == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['losses'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
            print("Epoch {}: Best Val {:.5f} | Best Test {:.5f}".format(epoch, best_val_loss, best_test_loss))
            with open(os.path.join("results", EXP_NAME, "losses.json"), "w") as outfile:
                json.dump(results, outfile, indent=4)
    return best_val_loss, best_test_loss, best_epoch

def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_count = 0
    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data
        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        optimizer.zero_grad()
        rows, cols = edges
        loc_dist = torch.sum((loc[cols] - loc[rows])**2, dim=1, keepdim=True)
        edge_attr = torch.cat([edge_attr, loc_dist], dim=1).detach()
        h_out, loc_pred, v_out = model(loc, vel, edges, edge_attr)
        loss = loss_mse(loc_pred, loc_end)
        if backprop:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch_size
        total_count += batch_size
    avg_loss = total_loss / total_count
    print("Epoch {} avg loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss

if __name__ == "__main__":
    main()
