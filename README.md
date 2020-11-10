# HybridPool

A Deep Hybrid Pooling Architecture for Graph Classification with Hierarchical Attention


This is a tensorflow based implementation of Hybrid Pooling as discussed in the paper.

Dataset:
1.  The dataset_graph folder contains all the datasets which we used in experiments of graph classification.


How to run: 
1) For Graph Classification: (Default dataset is set to MUTAG)
	python graph_classification.py



Requirements:
1) python (version 3.6 or above)
2) tensorflow (version 1.14)
3) networkx
4) keras
5) numpy
6) pickle
7) scipy
8) pandas
9) collections



Parameters:
1) For Graph Classification:
	dataset: The name of the dataset
	epoch: Number of epochs to train the model
	learning_rate: Learning rate
	embd_dim: Final Embedding dimension
	gcn_layer: Number of GCN layers
	gcn_dim: GCN Embedding dimension
	dropout: Dropout rate
	batch_size: Batch size


We can specify these parameters while running python file.
	For eg: To specify any other dataset, run following command: 
	python graph_classification.py --dataset NCI1

	

  
   

    
