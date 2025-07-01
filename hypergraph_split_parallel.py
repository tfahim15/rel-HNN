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
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
seed = 48
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
with open("hypergraph_data/"+dataset+'/features.pickle', 'rb') as f:
    features = pickle.load(f)
features = torch.tensor(features.toarray(), dtype=torch.float32)

with open("hypergraph_data/"+dataset+'/hypergraph.pickle', 'rb') as f:
    hypergraph = pickle.load(f)
    max_node  = max(max(values) for values in hypergraph.values())+1
    edges = hypergraph.keys()
    hypergraph_tensor = torch.zeros(len(edges), max_node)
    for i, edge in enumerate(edges):
        for j in hypergraph[edge]:
            hypergraph_tensor[(i,j)] = 1
    hypergraph = hypergraph_tensor

labels = []
for i in range(hypergraph_tensor.shape[0]):
    labels.append(0)
labels = torch.tensor(labels, dtype=torch.float32)

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
epochs = 100
k_folds = 5  # Number of Stratified K-Fold splits
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Store AUROC scores for each fold
auroc_scores = []

# Convert labels to NumPy for stratification
labels_np = labels.cpu().numpy()

        
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
    epochs = 1000
    
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    print(hypergraph_local.shape)
    print(features_local.shape)
    for epoch in range(epochs):
        V_0 = VNN0(features_local)
        partial = torch.matmul(hypergraph_local, V_0)
        dist.all_reduce(partial.clone(), op=dist.ReduceOp.SUM)

        E_0 = ENN0(partial)

        V_temp = VNN1(torch.matmul(torch.transpose(hypergraph_local, 0, 1), E_0))
        V_1 = VNN1(V_temp)

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
        #continue
        
        loss = criterion(E_fc[train_idx].view(-1), train_labels.view(-1).float().to(device))


        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        continue
        # Print and log memory usage
        #print_cuda_memory(rank)
        #with open("cuda_memory_log.txt", "a") as f:
            #f.write(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    torch.cuda.synchronize()

    #if rank == 0:
    dist.barrier()
    print(rank, "Multi-GPU Runtime: {:.2f} ms".format((time.time() - start_time) * 1000/epochs))
    return
    auroc = AUROC(task="binary").to(device)
    auroc_score = float(auroc(E_fc[train_test_idx[val_idx]].view(-1), val_labels.float().to(device)))
    print(auroc_score)
    dist.destroy_process_group()

def save_chunks(hypergraph, features, world_size, train_idx, val_idx):
    print("Saving chunks...")
    import os
    os.makedirs("chunks", exist_ok=True)

    # Split and save hypergraph chunks
    print(hypergraph.shape, features.shape)
    hypergraph_chunks = torch.chunk(hypergraph, world_size, dim=1)
    for i, chunk in enumerate(hypergraph_chunks ):
        torch.save(chunk, f"chunks/hypergraph_chunk_{i}.pt")

    # Split and save feature chunks
    features_chunks = torch.chunk(features, world_size, dim=0)
    for i, chunk in enumerate(features_chunks):
        
        torch.save(chunk, f"chunks/features_chunk_{i}.pt")

    # Save indices only
    torch.save(train_idx, "chunks/train_idx.pt")
    torch.save(val_idx, "chunks/val_idx.pt")



def main():
    # Stratified K-Fold Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_np, labels_np)):
        print(f"\nFold {fold+1}/{k_folds}")

        world_size = 1
        save_chunks(hypergraph, features, world_size, train_idx, val_idx)
        mp.spawn(parallel_train, args=(world_size,), nprocs=world_size)
        

if __name__ == "__main__":
#    mp.set_start_method("spawn", force=True)
    main()