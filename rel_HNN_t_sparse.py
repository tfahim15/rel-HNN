import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import AUROC
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def index_sparse_tensor(sparse_tensor, index_list):
    """
    Efficiently indexes a sparse tensor without converting it to dense.
    """
    index_mask = torch.isin(sparse_tensor.indices()[0], torch.tensor(index_list, device=sparse_tensor.device))
    new_indices = sparse_tensor.indices()[:, index_mask]
    new_values = sparse_tensor.values()[index_mask]

    # Adjust row indices to start from zero
    row_mapping = {old: new for new, old in enumerate(index_list)}
    new_indices[0] = torch.tensor([row_mapping[i.item()] for i in new_indices[0]], device=sparse_tensor.device)

    new_size = (len(index_list), sparse_tensor.shape[1])
    return torch.sparse_coo_tensor(new_indices, new_values, new_size).coalesce()


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
seed = 129
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
table_mapping = torch.load(dataset+"/table_mapping_tensor.pt").to(device)


# Define Model Class
class HNN(nn.Module):
    def __init__(self, inp, outp):
        super(HNN, self).__init__()
        self.linear = nn.Linear(inp, outp)
        self.activation_function = torch.sigmoid

    def forward(self, inputs):
        out = self.linear(inputs)
        return self.activation_function(out)


class TNN(nn.Module):
    def __init__(self, tn):
        super(TNN, self).__init__()
        self.W = nn.Parameter(torch.randn(tn, 1, requires_grad=True))  
        self.activation = nn.Softmax(dim=1)

    def forward(self, E, table_mapping):
        A = table_mapping @ self.W  
        E_f = torch.cat((E, A), dim=1)
        #E_f = self.activation(E_f)
        return E_f

# Model Initialization Parameters
EMBEDDING_DIM = 2
n_features = features.shape[1]
epochs =3000
k_folds = 5  # Number of Stratified K-Fold splits
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Store AUROC scores for each fold
auroc_scores = []

# Convert labels to NumPy for stratification
labels_np = labels.cpu().numpy()
train_test = train_test.coalesce() 
# Stratified K-Fold Cross-Validation Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(labels_np, labels_np)):
    print(f"\nFold {fold+1}/{k_folds}")

    # Convert indices back to tensors and move to device
    train_hypergraph = index_sparse_tensor(train_test, train_idx).to(device)
    val_hypergraph = index_sparse_tensor(train_test, val_idx).to(device)
    train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

    # Initialize fresh models for each fold
    VNN0 = HNN(n_features, EMBEDDING_DIM).to(device)
    TNN0 = TNN(table_mapping.shape[1]).to(device)
    ENN0 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN1 = HNN(EMBEDDING_DIM+1, EMBEDDING_DIM+1).to(device)
    ENN1 = HNN(EMBEDDING_DIM+1, EMBEDDING_DIM+1).to(device)
    VNN2 = HNN(EMBEDDING_DIM+1, EMBEDDING_DIM+1).to(device)
    FNN = HNN(EMBEDDING_DIM+1, 1).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)  # Binary Cross-Entropy Loss for binary classification
    optimizer = torch.optim.Adam(
        list(VNN0.parameters()) + list(ENN0.parameters()) +list(TNN0.parameters()) +
        list(VNN1.parameters()) + list(ENN1.parameters()) +
        list(VNN2.parameters()) + list(FNN.parameters()), lr=0.01)

    # AUROC metric
    auroc = AUROC(task="binary").to(device)

    # Training Loop
    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training Progress"):
        # Forward pass
        E = ENN0(torch.matmul(hypergraph, VNN0(features)))
        E = TNN0(E,table_mapping)
        
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
        E = TNN0(E,table_mapping)
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
print(auroc_scores)
print(f"\nFinal Stratified K-Fold Cross-Validation AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
