# Stanford Online Products Image Retrieval

Deep Learning based image retrieval system using ConvNeXt-Tiny and FAISS.

## Overview

This project explores image retrieval on the Stanford Online Products dataset.

The work is divided into multiple baselines:

### Baseline 1: Classification-based Retrieval

A ConvNeXt-Tiny model is fine-tuned as a classification model using the product class IDs.

- Number of classes: 11,318
- Objective: Cross Entropy Classification
- Backbone: ConvNeXt-Tiny

After training, the classification head is removed and the backbone is used as a feature extractor.

The extracted feature embeddings are indexed using FAISS for nearest neighbor search.

Retrieval indexes:

- IndexFlatL2 (initial baseline)
- IndexIVFFlat (large-scale search)


### Baseline 2: Metric Learning Retrieval (Future Work)

The second baseline will focus on learning a more discriminative embedding space using metric learning.

Planned approach:

- Backbone: ConvNeXt-Tiny
- Loss: Triplet Loss
- Retrieval Index: FAISS HNSW

The goal is to optimize the embedding space so that visually similar products are closer together while dissimilar products are separated.

## Pipeline

### Baseline 1  

Image
|
v
ConvNeXt-Tiny
|
v
Classification Head
|
v
Feature Extraction
|
v
FAISS IndexFlat / IVF
|
v
Top-K Similar Images



### Baseline 2


Image
|
v
ConvNeXt-Tiny
|
v
Embedding Vector
|
v
Triplet Loss Training
|
v
FAISS HNSW
|
v
Nearest Neighbor Retrieval




## Dataset

Stanford Online Products Dataset

- Product image retrieval benchmark
- 11,318 training classes


## Technologies

- PyTorch
- Torchvision
- ConvNeXt-Tiny
- FAISS
- Metric Learning


## Future Improvements

- Evaluate Recall@K
- Evaluate mAP
- Hyperparameter tuning for FAISS indexes
