import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
seed = 48
torch.manual_seed(seed)
np.random.seed(seed)

# Parse dataset argument
parser = argparse.ArgumentParser(description="Read dataset from file path provided as argument.")
parser.add_argument('--dataset', help="Enable dataset.")
args = parser.parse_args()
dataset = args.dataset

# Load tensors and move to device
features = torch.load(dataset+'/feature_tensor.pt').to(device)
hypergraph = torch.load(dataset+"/hypergraph_tensor.pt").to(device)
train_test = torch.load(dataset+"/train_test_tensor.pt").to(device)
labels = torch.load(dataset+"/labels_tensor.pt").to(device)

# Define Model Class
class HNN(nn.Module):
    def __init__(self, inp, outp):
        super(HNN, self).__init__()
        self.linear = nn.Linear(inp, outp)
    
    def forward(self, inputs):
        return self.linear(inputs)

# Model Initialization Parameters
EMBEDDING_DIM = 2
n_features = features.shape[1]
epochs = 10000
k_folds = 5  # Number of K-Fold splits
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Store RMSE scores for each fold
rmse_scores = []

# Convert labels to NumPy for stratification
labels_np = labels.cpu().numpy()

# K-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(kf.split(train_test.cpu().numpy())):
    print(f"\nFold {fold+1}/{k_folds}")

    # Convert indices back to tensors and move to device
    train_hypergraph, val_hypergraph = train_test[train_idx].to(device), train_test[val_idx].to(device)
    train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

    # Initialize fresh models for each fold
    VNN0 = HNN(n_features, EMBEDDING_DIM).to(device)
    ENN0 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN1 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    ENN1 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN2 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    FNN = HNN(EMBEDDING_DIM, 1).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss().to(device)  # Mean Squared Error Loss for regression
    optimizer = torch.optim.Adam(
        list(VNN0.parameters()) + list(ENN0.parameters()) +
        list(VNN1.parameters()) + list(ENN1.parameters()) +
        list(VNN2.parameters()) + list(FNN.parameters()), lr=0.01)

    # Training Loop
    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training Progress"):
        # Forward pass
        E = ENN0(torch.matmul(hypergraph, VNN0(features)))
        V = VNN1(torch.matmul(torch.transpose(hypergraph, 0, 1), E))
        
        E = ENN1(torch.matmul(hypergraph, VNN1(V)))
        V = VNN2(torch.matmul(torch.transpose(hypergraph, 0, 1), E))

        E = FNN(torch.matmul(train_hypergraph, V))

        # Compute loss
        loss = criterion(E.view(-1), train_labels.float())
        if epoch%1000==0:
            print(epoch, loss)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    ENN0.eval()
    VNN0.eval()
    ENN1.eval()
    VNN1.eval()
    VNN2.eval()
    FNN.eval()

    with torch.no_grad():
        E = ENN0(torch.matmul(hypergraph, VNN0(features)))
        V = VNN1(torch.matmul(torch.transpose(hypergraph, 0, 1), E))
        
        E = ENN1(torch.matmul(hypergraph, VNN1(V)))
        V = VNN2(torch.matmul(torch.transpose(hypergraph, 0, 1), E))

        E = FNN(torch.matmul(val_hypergraph, V))
        rmse_score = torch.sqrt(criterion(E.view(-1), val_labels.float())).item()

    print(f"Fold {fold+1} RMSE: {rmse_score}")
    rmse_scores.append(rmse_score)

# Compute and print final cross-validation results
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
print(f"\nFinal K-Fold Cross-Validation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
