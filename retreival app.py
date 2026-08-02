# Generated from: retreival app.ipynb
# Converted at: 2026-08-02T15:57:47.272Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from PIL import Image

import numpy as np
import faiss

from huggingface_hub import hf_hub_download


# ==========================
# Streamlit Config
# ==========================

st.set_page_config(
    page_title="Stanford Products Retrieval",
    layout="wide"
)


# ==========================
# Device
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================
# Hugging Face Repos
# ==========================

ASSET_REPO = "heshamdahy/stanford-products-retrieval-assets"

IMAGE_REPO = "heshamdahy/stanford-products-images"



# ==========================
# OpenCLIP Model Class
# لازم تكون موجودة لأن الموديل محفوظ كامل
# ==========================

class open_clip_model(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.clip = model

        self.ffn = nn.Sequential(
            nn.Linear(512,256)
        )


    def forward(self,x):

        x = self.clip.encode_image(x)

        x = self.ffn(x)

        x = F.normalize(
            x,
            dim=1
        )

        return x



# ==========================
# Load Assets
# ==========================

@st.cache_resource
def load_assets():


    # Model

    model_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="open_clip.model"
    )


    # FAISS HNSW

    index_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="index_hnsw.faiss"
    )


    # Paths

    paths_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="image_paths_relative.npy"
    )



    # Load model

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )


    model.to(device)

    model.eval()



    # Load FAISS

    index = faiss.read_index(
        index_path
    )


    # Load image paths

    image_paths = np.load(
        paths_path,
        allow_pickle=True
    )


    return (
        model,
        index,
        image_paths
    )



model, index, image_paths = load_assets()



# ==========================
# Image Transform
# ==========================

transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])



# ==========================
# Feature Extraction
# ==========================

def extract_feature(image):


    image = transform(image)


    image = image.unsqueeze(0)


    image = image.to(device)



    with torch.no_grad():

        embedding = model(image)



    embedding = embedding.cpu().numpy()


    return embedding.astype(
        "float32"
    )



# ==========================
# Download Image Cache
# ==========================

@st.cache_data
def load_image_from_hf(path):


    image_file = hf_hub_download(

        repo_id=IMAGE_REPO,

        repo_type="dataset",

        filename=path
    )


    image = Image.open(
        image_file
    ).convert("RGB")


    return image



# ==========================
# UI
# ==========================

st.title(
    "🛍️ Stanford Products Image Retrieval"
)


st.write(
    "Upload an image and retrieve similar products using OpenCLIP + FAISS HNSW"
)



uploaded_file = st.file_uploader(

    "Upload product image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



k = st.slider(

    "Number of retrieved images",

    min_value=1,

    max_value=20,

    value=5

)



if uploaded_file:


    query_image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.subheader(
        "Query Image"
    )


    st.image(
        query_image,
        width=300
    )



    if st.button("Search"):


        query_embedding = extract_feature(
            query_image
        )



        distances, indices = index.search(

            query_embedding,

            k

        )



        st.subheader(
            "Retrieved Images"
        )



        cols = st.columns(k)



        for i, idx in enumerate(indices[0]):


            image_path = str(
                image_paths[idx]
            )


            try:


                result_image = load_image_from_hf(
                    image_path
                )


                with cols[i]:

                    st.image(

                        result_image,

                        caption=f"Distance: {distances[0][i]:.4f}",

                        use_container_width=True

                    )


            except Exception as e:


                with cols[i]:

                    st.error(
                        f"Failed loading image\n{image_path}\n{e}"
                    )