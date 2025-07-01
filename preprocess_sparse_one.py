import argparse
import pandas as pd
import glob
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Read dataset from file path provided as argument.")
parser.add_argument('--dataset', help="Enable dataset.")
parser.add_argument('--target_file', help="Enable target file.")
parser.add_argument('--target_column', help="Enable target column.") 

args = parser.parse_args()

dataset = args.dataset
target_file = args.target_file
target_column = str(args.target_column) 


files = glob.glob(dataset+"/*.csv")
v_id = dict()
for f in files:
    line = open(f).readlines()[0]
    attrs = line.replace("\n", "").split(",")
    for attr in attrs:
        if attr == target_column and target_file in f:
            continue
        if attr not in v_id:
            v_id[attr] = dict()


i = 0
for f in files:
    lines = open(f).readlines()
    attrs = lines[0].replace("\n", "").split(",")
    lines = lines[1:]
    for line in lines:
        values = line.replace("\n", "").split(",")
        for v in range(len(values)):
            try:
                if attrs[v] not in v_id:
                    continue
                if values[v] not in v_id[attrs[v]]:
                    v_id[attrs[v]][values[v]] = i
                    i += 1
            except:
                print(f)

Hypergraph_data = []
table_id = []
train_test_idx = []
for f in range(len(files)):
    tid = f
    f = files[f]
    lines = open(f).readlines()
    attrs = lines[0].replace("\n", "").split(",")
    lines = lines[1:]
    for line in lines:
        values = line.replace("\n", "").split(",")
        edge = []
        for v in range(len(values)):
            if attrs[v] not in v_id:
                continue
            edge.append(v_id[attrs[v]][values[v]])
        if target_file in f:
            train_test_idx.append(len(Hypergraph_data))
        Hypergraph_data.append(edge)
        table_id.append(tid)


train_test_data = []
labels = []
lines = open(dataset+"/"+target_file+".csv").readlines()
attrs = lines[0].replace("\n", "").split(",")
lines = lines[1:]
for line in lines:
    values = line.replace("\n", "").split(",")
    edge = []
    label = None
    for v in range(len(values)):
        if attrs[v] not in v_id: 
            label = values[v].replace("\"", "")
            continue
        edge.append(v_id[attrs[v]][values[v]])
    train_test_data.append(edge)
    labels.append(label)

unique_labels = np.unique(labels)  # Sorted unique labels
label_mapping = {label: float(idx) for idx, label in enumerate(unique_labels)}
labels_ = [label_mapping[l] for l in labels]
labels = labels_

# Data from the file, each line as a list of integers


max_tid = max(table_id) + 1
table_mapping_tensor = torch.zeros((len(table_id), max_tid), dtype=torch.float32)
for i, tid in enumerate(table_id):
    table_mapping_tensor[i, tid] = 1

torch.save(table_mapping_tensor, dataset+'/table_mapping_tensor.pt')

# Determine the size of the tensor (max index + 1)
max_index = max(max(line) for line in Hypergraph_data) + 1

# Create the tensor
indices = []
values = []

for i, edge in enumerate(Hypergraph_data):
    for node in edge:
        indices.append([i, node])  # Row index is the hyperedge, column index is the node
        values.append(1.0)  # Binary representation

# Convert lists to tensors
indices = torch.tensor(indices, dtype=torch.long).t()  # Transpose to match sparse format
values = torch.tensor(values, dtype=torch.float32)

# Create sparse tensor
hypergraph_tensor = torch.sparse_coo_tensor(indices, values, (len(Hypergraph_data), max_index))

torch.save(hypergraph_tensor, dataset+'/hypergraph_tensor.pt')

indices = []
values = []
for i, row in enumerate(train_test_data):
    for col in row:
        indices.append([i, col])  # (row, column)
        values.append(1.0)  # Binary indicator

indices = torch.tensor(indices, dtype=torch.long).T  # Transpose to shape (2, N)
values = torch.tensor(values, dtype=torch.float32)

train_test_tensor = torch.sparse_coo_tensor(indices, values, (len(train_test_data), max_index))

labels_tensor = torch.tensor(labels, dtype=torch.float32)

torch.save(train_test_tensor, dataset+'/train_test_tensor.pt')
torch.save(labels_tensor, dataset+'/labels_tensor.pt')

indices = torch.arange(max_index).repeat(2, 1)  # Row and column indices for diagonal
values = torch.ones(max_index, dtype=torch.float32)  # Diagonal values are 1
feature_tensor = torch.sparse_coo_tensor(indices, values, (max_index, max_index))

torch.save(feature_tensor, dataset+'/feature_tensor.pt')


torch.save(train_test_idx, dataset+'/train_test_idx.pt')