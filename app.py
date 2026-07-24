# Generated from: app.ipynb
# Converted at: 2026-07-24T22:09:36.798Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import streamlit as st

import torch
import torch.nn as nn

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
# Config
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


repo_id = "heshamdahy/stanford-products-retrieval-assets"



# ==========================
# Load Assets
# ==========================

@st.cache_resource
def load_assets():


    # Download files

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pth"
    )


    index_path = hf_hub_download(
        repo_id=repo_id,
        filename="index.faiss"
    )


    embeddings_path = hf_hub_download(
        repo_id=repo_id,
        filename="embeddings.npy"
    )


    image_paths_path = hf_hub_download(
        repo_id=repo_id,
        filename="image_paths.npy"
    )



    # ======================
    # Load ConvNeXt Model
    # ======================

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )


    # Remove classifier
    # keep only feature extractor

    if hasattr(model, "classifier"):

        model.classifier = nn.Identity()



    model.to(device)

    model.eval()



    # ======================
    # Load FAISS
    # ======================

    index = faiss.read_index(
        index_path
    )



    # ======================
    # Load Embeddings
    # ======================

    embeddings = np.load(
        embeddings_path
    )



    # ======================
    # Load Image Paths
    # ======================

    image_paths = np.load(
        image_paths_path,
        allow_pickle=True
    )


    return (
        model,
        index,
        embeddings,
        image_paths
    )



model, index, embeddings, image_paths = load_assets()



# ==========================
# Debug Information
# ==========================

st.sidebar.write(
    "FAISS dimension:",
    index.d
)

st.sidebar.write(
    "Embedding shape:",
    embeddings.shape
)



# ==========================
# Transform
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


    image = transform(
        image
    )


    image = image.unsqueeze(0)


    image = image.to(device)



    with torch.no_grad():

        feature = model(
            image
        )



    # ConvNeXt output:
    # [1,768,1,1]

    if feature.ndim == 4:

        feature = torch.flatten(
            feature,
            start_dim=1
        )



    feature = feature.cpu().numpy()



    # L2 Normalization

    feature = feature / np.linalg.norm(
        feature,
        axis=1,
        keepdims=True
    )


    return feature.astype(
        "float32"
    )



# ==========================
# UI
# ==========================

st.title(
    "🛍️ Stanford Products Image Retrieval"
)


st.write(
    "ConvNeXt-Tiny + FAISS"
)



uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



k = st.slider(
    "Number of results",
    min_value=1,
    max_value=20,
    value=5
)



if uploaded_file:


    query_image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )


    st.subheader(
        "Query Image"
    )


    st.image(
        query_image,
        width=300
    )



    if st.button(
        "Search"
    ):


        query_embedding = extract_feature(
            query_image
        )


        # Check dimensions

        if query_embedding.shape[1] != index.d:

            st.error(
                f"""
                Feature dimension mismatch!

                FAISS expects: {index.d}

                Model output: {query_embedding.shape[1]}
                """
            )

            st.stop()



        # Search

        distances, indices = index.search(
            query_embedding,
            k
        )



        st.subheader(
            "Similar Products"
        )


        cols = st.columns(k)



        for i, idx in enumerate(indices[0]):


            path = image_paths[idx]


            try:

                result = Image.open(
                    path
                ).convert(
                    "RGB"
                )


                with cols[i]:

                    st.image(
                        result,
                        caption=f"Distance: {distances[0][i]:.4f}"
                    )


            except Exception as e:


                with cols[i]:

                    st.error(
                        str(e)
                    )
