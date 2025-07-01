import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import AUROC
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
seed = 129
torch.manual_seed(seed)
np.random.seed(seed)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12335'

# Parse dataset argument
parser = argparse.ArgumentParser(description="Read dataset from file path provided as argument.")
parser.add_argument('--dataset', help="Enable dataset.")
args = parser.parse_args()
dataset = args.dataset

# Load tensors and move to device
features = torch.load(dataset+'/feature_tensor.pt')#.to(device)
hypergraph = torch.load(dataset+"/hypergraph_tensor.pt")#.to(device)
train_test_idx = torch.load(dataset+"/train_test_idx.pt")#.to(device)
train_test_idx = np.array(train_test_idx)
train_test = torch.load(dataset+"/train_test_tensor.pt")
labels = torch.load(dataset+"/labels_tensor.pt")#.to(device)

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
epochs = 1000
k_folds = 5  # Number of Stratified K-Fold splits
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Store AUROC scores for each fold
auroc_scores = []

# Convert labels to NumPy for stratification
labels_np = labels.cpu().numpy()

def print_cuda_memory(rank):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        allocated = torch.cuda.memory_allocated(device) / 1024**2  
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

        print(f"[GPU {rank}] Allocated: {allocated:.2f} MB")
        print(f"[GPU {rank}] Reserved:  {reserved:.2f} MB")
        print(f"[GPU {rank}] Max Allocated: {max_allocated:.2f} MB")
        print(f"[GPU {rank}] Max Reserved:  {max_reserved:.2f} MB")
    else:
        print("CUDA is not available.")
        
def parallel_train(rank, world_size):
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print(rank, hypergraph.get_device())
    
    hypergraph_local = torch.load(f"chunks/hypergraph_chunk_{rank}.pt").to(device)
    features_local = torch.load(f"chunks/features_chunk_{rank}.pt").to(device)
    
    train_idx = torch.load("chunks/train_idx.pt") 
    val_idx = torch.load("chunks/val_idx.pt") 
    
    train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)
    
    VNN0 = HNN(n_features, EMBEDDING_DIM).to(device)
    ENN0 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN1 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    ENN1 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    VNN2 = HNN(EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    FNN = HNN(EMBEDDING_DIM, 1).to(device)
    
    VNN0 = nn.parallel.DistributedDataParallel(VNN0, device_ids=[rank])
    ENN0 = nn.parallel.DistributedDataParallel(ENN0, device_ids=[rank])
    VNN1 = nn.parallel.DistributedDataParallel(VNN1, device_ids=[rank])
    ENN1 = nn.parallel.DistributedDataParallel(ENN1, device_ids=[rank])
    VNN2 = nn.parallel.DistributedDataParallel(VNN2, device_ids=[rank])
    FNN = nn.parallel.DistributedDataParallel(FNN, device_ids=[rank])

    
    criterion = nn.CrossEntropyLoss().to(device)  # Binary Cross-Entropy Loss for binary classification
    optimizer = torch.optim.Adam(
        list(VNN0.parameters()) + list(ENN0.parameters()) +
        list(VNN1.parameters()) + list(ENN1.parameters()) +
        list(VNN2.parameters()) + list(FNN.parameters()), lr=0.01)

    #start.record()
    epochs = 5#3000
    
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    print(hypergraph_local.shape)
    print(features_local.shape)
    for epoch in range(epochs):
        splits = 20
        
        features_local_splits = chunk_sparse_tensor(features_local, splits, dim=0)
        hypergraph_local_splits = chunk_sparse_tensor(hypergraph_local, splits, dim=1)
        
        partial = torch.zeros(hypergraph_local.shape[0],EMBEDDING_DIM, device=device, dtype=features_local.dtype)

        for i in range(splits):
                feat = features_local_splits[i].to_dense().to(device)
                V_0 = VNN0(feat)
                h_local = hypergraph_local_splits[i].to_dense().to(device)
                partial = partial + torch.matmul(h_local, V_0).detach()
            
        
        dist.all_reduce(partial.clone(), op=dist.ReduceOp.SUM)

        E_0 = ENN0(partial)

        V_temp = VNN1(torch.matmul(torch.transpose(hypergraph_local, 0, 1), E_0))
        V_1 = VNN1(V_temp)
        
        V_1_splits = torch.chunk(V_1, splits, dim=0)
        E_temp = torch.matmul(hypergraph_local, V_1)
        E_1 = ENN1(E_temp)
        dist.all_reduce(E_1.clone(), op=dist.ReduceOp.SUM)

        H_t = torch.transpose(hypergraph_local, 0, 1)
        V_2 = torch.matmul(H_t, E_1)
        V_3 = VNN2(V_2)

        E_2 = torch.matmul(hypergraph_local, V_3)
        E_final = FNN(E_2)
        E_fc = E_final.clone()
        dist.all_reduce(E_fc, op=dist.ReduceOp.SUM)
        loss = criterion(E_fc[train_test_idx[train_idx]].view(-1), train_labels.float().to(device))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    torch.cuda.synchronize()

    #if rank == 0:
    dist.barrier()
    print(rank, "Multi-GPU Runtime: {:.2f} ms".format((time.time() - start_time) * 1000/epochs))
    auroc = AUROC(task="binary").to(device)
    auroc_score = float(auroc(E_fc[train_test_idx[val_idx]].view(-1), val_labels.float().to(device)))
    print(auroc_score)
    dist.destroy_process_group()

def chunk_sparse_tensor(sparse_tensor, chunks, dim=0):
    assert sparse_tensor.is_sparse, "Input must be a sparse tensor"
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    size = sparse_tensor.size(dim)

    # Compute chunk sizes
    chunk_sizes = torch.tensor_split(torch.arange(size), chunks)

    chunked_tensors = []
    for chunk in chunk_sizes:
        # Select entries in the current chunk
        mask = (indices[dim] >= chunk[0]) & (indices[dim] < chunk[-1] + 1)
        new_indices = indices[:, mask].clone()
        new_indices[dim] -= chunk[0]  # Shift indices to local chunk coordinates
        new_values = values[mask]

        # New size for this chunk
        new_size = list(sparse_tensor.size())
        new_size[dim] = len(chunk)

        chunked_tensors.append(
            torch.sparse_coo_tensor(new_indices, new_values, size=new_size)
        )

    return chunked_tensors

def save_chunks(hypergraph, features, world_size, train_idx, val_idx):
    print("Saving chunks...")
    import os
    os.makedirs("chunks", exist_ok=True)

    # Split and save hypergraph chunks
    hypergraph_chunks = chunk_sparse_tensor(hypergraph, world_size, dim=1)
    for i, chunk in enumerate(hypergraph_chunks ):
        torch.save(chunk, f"chunks/hypergraph_chunk_{i}.pt")

    # Split and save feature chunks
    features_chunks = chunk_sparse_tensor(features, world_size, dim=0)
    for i, chunk in enumerate(features_chunks):
        torch.save(chunk, f"chunks/features_chunk_{i}.pt")

    # Save indices only
    torch.save(train_idx, "chunks/train_idx.pt")
    torch.save(val_idx, "chunks/val_idx.pt")




def main():
    # Stratified K-Fold Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_np, labels_np)):
        print(f"\nFold {fold+1}/{k_folds}")

        world_size = 2
        save_chunks(hypergraph, features, world_size, train_idx, val_idx)
        mp.spawn(parallel_train, args=(world_size,), nprocs=world_size)
        

if __name__ == "__main__":
#    mp.set_start_method("spawn", force=True)
    main()
