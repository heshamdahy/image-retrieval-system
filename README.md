# Stanford Online Products Image Retrieval

A deep learning-based image retrieval system built on the **Stanford Online Products** dataset. This project explores both **classification-based** and **metric learning** approaches for large-scale visual similarity search using **FAISS**.

---

# Overview

The project is divided into two baselines, each representing a different retrieval strategy.

## Baseline 1: Classification-Based Retrieval

The first baseline fine-tunes a **ConvNeXt-Tiny** model as a product classifier.

* **Backbone:** ConvNeXt-Tiny
* **Objective:** Cross Entropy Loss
* **Training Classes:** 11,318

After training, the classification head is removed and the backbone is used as a feature extractor.

The extracted embeddings are indexed using FAISS for image retrieval.

### Retrieval Indexes

* FAISS IndexFlatL2
* FAISS IndexIVFFlat

---

## Baseline 2: Metric Learning Retrieval (Final)

The second baseline replaces the classification objective with **Metric Learning**, enabling the model to learn a more discriminative embedding space for visual similarity.

### Architecture

* **Backbone:** OpenCLIP
* **Embedding Head:** Fully Connected Layer (512 → 256)
* **Loss Function:** Triplet Loss
* **Embedding Normalization:** L2 Normalization
* **Similarity Metric:** Cosine Similarity
* **Retrieval Index:** FAISS HNSW

Instead of predicting product classes, the model learns embeddings where:

* Similar products are mapped close together.
* Different products are pushed farther apart.

This approach significantly improves image retrieval quality compared to the classification baseline.

---

# Pipeline

## Baseline 1

```text
Image
   │
   ▼
ConvNeXt-Tiny
   │
   ▼
Classification Head
   │
   ▼
Feature Extraction
   │
   ▼
FAISS IndexFlatL2 / IndexIVFFlat
   │
   ▼
Top-K Similar Images
```

## Baseline 2

```text
Image
   │
   ▼
OpenCLIP
   │
   ▼
Projection Head (512 → 256)
   │
   ▼
Triplet Loss Training
   │
   ▼
L2 Normalization
   │
   ▼
FAISS HNSW (Cosine Similarity)
   │
   ▼
Top-K Similar Images
```

---

# Features

* Two complete retrieval baselines
* ConvNeXt-Tiny classification-based retrieval
* OpenCLIP metric learning retrieval
* Triplet Loss training
* FAISS IndexFlatL2
* FAISS IndexIVFFlat
* FAISS HNSW indexing
* Cosine Similarity search
* Streamlit web application
* Hugging Face model hosting

---

# Live Demo

**Streamlit App**

https://image-retrieval-system-3ucsgpxujhcq4a2atjffaj.streamlit.app/

---

# Dataset

**Stanford Online Products**

https://www.kaggle.com/datasets/liucong12601/stanford-online-products-dataset

* Product image retrieval benchmark
* 11,318 training classes
* Large-scale fine-grained product recognition dataset

---

# Technologies

* Python
* PyTorch
* Torchvision
* OpenCLIP
* ConvNeXt-Tiny
* FAISS
* Metric Learning
* Triplet Loss
* Streamlit
* Hugging Face Hub

---

# Model Assets

Due to GitHub file size limitations, the trained models and retrieval assets are hosted on Hugging Face.

Assets include:

* OpenCLIP model
* ConvNeXt-Tiny model
* FAISS indexes
* Image embeddings
* Image path mappings

**Hugging Face**

https://huggingface.co/heshamdahy/stanford-products-retrieval-assets

---

# Repository

GitHub Repository

https://github.com/heshamdahy/image-retrieval-system

---

# Future Work

* Evaluate Recall@K
* Evaluate mAP
* Compare additional metric learning losses (ArcFace, Circle Loss, Contrastive Loss)
* Experiment with larger Vision-Language Models
* Improve retrieval latency on million-scale datasets
