import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import AUROC
from sklearn.model_selection import StratifiedKFold
import numpy as np

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
        self.activation_function = torch.sigmoid

    def forward(self, inputs):
        out = self.linear(inputs)
        return self.activation_function(out)


# Model Initialization Parameters
EMBEDDING_DIM = 2
n_features = features.shape[1]
epochs = 400
k_folds = 5  # Number of Stratified K-Fold splits
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Store AUROC scores for each fold
auroc_scores = []

# Convert labels to NumPy for stratification
labels_np = labels.cpu().numpy()

# Stratified K-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(train_test.cpu().numpy(), labels_np)):
    print(f"\nFold {fold+1}/{k_folds}")

    # Convert indices back to tensors and move to device
    train_hypergraph, val_hypergraph = train_test[train_idx].to(device), train_test[val_idx].to(device)
    train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

    # Initialize fresh models for each fold
    VNN0 = HNN(n_features, EMBEDDING_DIM).to(device)
    ENN0 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN1 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN12 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    ENN1 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN2 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    FNN = HNN(EMBEDDING_DIM, 1).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)  # Binary Cross-Entropy Loss for binary classification
    optimizer = torch.optim.Adam(
        list(VNN0.parameters()) + list(ENN0.parameters()) +
        list(VNN1.parameters()) + list(ENN1.parameters()) +
        list(VNN2.parameters()) + list(FNN.parameters()), lr=0.01)

    # AUROC metric
    auroc = AUROC(task="binary").to(device)

    # Training Loop
    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training Progress"):
        # Forward pass
        E = ENN0(torch.matmul(hypergraph, VNN0(features)))
        V = VNN1(torch.matmul(torch.transpose(hypergraph, 0, 1), E))
        
        E = ENN1(torch.matmul(hypergraph, VNN1(V)))
        V = VNN2(torch.matmul(torch.transpose(hypergraph, 0, 1), E))

        E = FNN(torch.matmul(train_hypergraph, V))

        # Compute loss
        loss = criterion(E.view(-1), train_labels.float())  # BCE requires float labels
        #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

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
        auroc_score = float(auroc(E.view(-1), val_labels.float()))

    print(f"Fold {fold+1} AUROC: {auroc_score}")
    auroc_scores.append(auroc_score)

# Compute and print final cross-validation results
mean_auroc = np.mean(auroc_scores)
std_auroc = np.std(auroc_scores)
print(f"\nFinal Stratified K-Fold Cross-Validation AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
