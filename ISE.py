# Generated from: ISE.ipynb
# Converted at: 2026-07-24T20:32:54.286Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import streamlit as st
import torch
import faiss
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# ----------------------------
# Load model
# ----------------------------
model = torch.load("model.pth", map_location="cpu")
model.eval()

# ----------------------------
# Load FAISS
# ----------------------------
index = faiss.read_index("index.faiss")

embeddings = np.load("embeddings.npy")
image_paths = np.load("image_paths.npy", allow_pickle=True)

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ----------------------------
# Feature Extractor
# ----------------------------
feature_extractor = torch.nn.Sequential(
    model.features,
    model.avgpool,
    model.classifier[:-1]
)

# ----------------------------
# UI
# ----------------------------
st.title("Image Retrieval")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","png","jpeg"]
)

k = st.slider(
    "Number of retrieved images",
    1,
    20,
    5
)

if uploaded is not None:

    img = Image.open(uploaded).convert("RGB")

    st.image(img,width=250)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        feature = feature_extractor(x)

    feature = feature.numpy().astype("float32")

    D,I = index.search(feature,k)

    st.write("Results")

    cols = st.columns(k)

    for i,col in enumerate(cols):

        result = Image.open(image_paths[I[0][i]])

        col.image(result,use_container_width=True)