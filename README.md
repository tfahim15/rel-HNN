# Rel-HNN: Split Parallel Hypergraph Neural Network for Learning on Relational Databases

Rel-HNN is a novel framework designed to learn from relational databases by transforming them into hypergraphs and leveraging a split-parallel training paradigm to scale hypergraph neural networks (HNNs) efficiently. The approach preserves both intra- and inter-tuple relationships and supports training over large-scale, multi-relational tabular data.

## 🔗 Dataset Downloads

To reproduce the experiments from our paper, you can download the datasets from the following link:

📁 [Download Datasets](https://drive.google.com/file/d/1dkcFUda7ar-CF7Kxz7WAvFF-djk-4Bsy/view?usp=sharing)  

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch 
- NumPy
- Scikit-learn

## 📜 Preprocessing Scripts

- `preprocess_one.py`: Preprocess with one-hot encoding-based features
- `preprocess_av.py`: Preprocess with attribute-value features
- `preprocess_reg_one.py`: Preprocess regression datasets with one-hot encoding
- `preprocess_reg_av.py`: Preprocess regression datasets with attribute-value features
- `preprocess_sparse_one.py`: Preprocess with one-hot encoding-based features for sparse hypergraphs
- `preprocess_sparse_av.py`: Preprocess with attribute-value features for sparse hypergraphs

### ▶️ Example Usage

To preprocess a dataset (e.g., **Cora**) using `preprocess_one.py`, run:

```bash
python3 preprocess_one.py --dataset Cora --target_file paper --target_column label
```
### ⚙️ Argument Descriptions

| Argument | Description |
|----------|-------------|
| --dataset | **(Required)** Name of the dataset directory. This folder should contain all CSV files (or similar formats) that represent tables in your relational database. For example, `Cora`. |
| --target_file | **(Required)** Name of the file (without `.csv` extension) that contains the column to be predicted. For instance, `paper` means `paper.csv` is the target table with labels. |
| --target_column | **(Required)** Name of the target column in the `target_file`. This is the column your model will be trained to predict, e.g., `label`. |


## 🧠 rel-HNN Scripts

- `rel_HNN.py`: Standard classification model using Relational Hypergraph Neural Network (rel-HNN)
- `rel_HNN_t.py`: Rel-HNN with table embedding (rel-HNN-t)
- `rel_HNN_mc.py`: Multi-class classification version of rel-HNN
- `rel_HNN_t_mc.py`: Multi-class classification version of rel-HNN-t
- `rel_HNN_reg.py`: Regression version of rel-HNN
- `rel_HNN_reg_t.py`: Regression version of rel-HNN-t
- `rel_HNN_sparse.py`: Rel-HNN with support for sparse hypergraphs
- `rel_HNN_t_sparse.py`: Rel-HNN-t with support for sparse hypergraphs
  
### ▶️ Example Command

```bash
python3 rel_HNN.py --dataset Cora
```

| Argument           | Description |
|--------------------|-------------|
| --dataset        | **(Required)** Name of the dataset directory. This folder should contain all CSV files (or similar formats) that represent tables in your relational database. For example, `Cora`. |



## ⚙️ Split-Parallel rel-HNN Scripts

- `split_parallel_rel_HNN.py`: Parallel training of Rel-HNN
- `split_parallel_rel_HNN_sparse.py`: Parallel training on sparse hypergraphs
### ▶️ Example Command

```bash
python3 split_parallel_rel_HNN.py --dataset Cora
```

| Argument           | Description |
|--------------------|-------------|
| --dataset        | **(Required)** Name of the dataset directory. This folder should contain all CSV files (or similar formats) that represent tables in your relational database. For example, `Cora`. |



## 🧪 Split-Parallel Hypergraph Neural Networks Scripts

- `hypergraph_split_parallel.py`: Split-parallel training on real hypergraph data
- `hypergraph_split_parallel_synthetic.py`: Split-parallel training on synthetic hypergraph data

### ▶️ Example Command

```bash
python3 hypergraph_split_parallel.py --dataset dblp
```

| Argument           | Description |
|--------------------|-------------|
| --dataset        | **(Required)** Name of the dataset directory. This folder should contain all CSV files (or similar formats) that represent tables in your relational database. For example, `dblp`. |


