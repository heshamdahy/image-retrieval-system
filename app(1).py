
import streamlit as st

import torch
import torchvision.transforms as transforms
import torch.nn as nn
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
# Configuration
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ضع اسم Hugging Face repo الخاص بك هنا
repo_id = "heshamdahy/stanford-products-retrieval-assets"



# ==========================
# Load Model + Assets
# ==========================

@st.cache_resource
def load_assets():

    # Download files from Hugging Face

    model_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.pth"
    )


    index_file = hf_hub_download(
        repo_id=repo_id,
        filename="index.faiss"
    )


    embeddings_file = hf_hub_download(
        repo_id=repo_id,
        filename="embeddings.npy"
    )


    image_paths_relative = hf_hub_download(
        repo_id=repo_id,
        filename="image_paths_relative.npy"
    )



    # ======================
    # Load Model
    # ======================

    model = torch.load(
        model_file,
        map_location=device,
        weights_only=False
    )
    
    model.classifier = nn.Identity()

    model.to(device)
    model.eval()



    # ======================
    # Load FAISS Index
    # ======================

    index = faiss.read_index(
        index_file
    )



    # ======================
    # Load Embeddings
    # ======================

    embeddings = np.load(
        embeddings_file
    )



    # ======================
    # Load Image Paths
    # ======================

    image_paths = np.load(
        image_paths_relative,
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
# Image Preprocessing
# ==========================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
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

        feature = model(image)



    # ConvNeXt output:
    # [batch, 768, 1, 1]

    if feature.ndim == 4:

        feature = torch.flatten(
            feature,
            start_dim=1
        )



    feature = feature.cpu().numpy()



    # Normalize embedding

    feature = feature / np.linalg.norm(
        feature,
        axis=1,
        keepdims=True
    )


    return feature.astype(
        "float32"
    )



# ==========================
# Streamlit UI
# ==========================

st.title(
    "🛍️ Stanford Products Image Retrieval"
)


st.write(
    "Upload an image and retrieve similar products using ConvNeXt + FAISS"
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
    "Number of similar images",
    min_value=1,
    max_value=20,
    value=5
)



# Dataset repo containing all images
IMAGE_REPO_ID = "heshamdahy/stanford-products-images"
repo_id = "heshamdahy/stanford-products-retrieval-assets"

if uploaded_file:

    query_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Query Image")
    st.image(query_image, width=300)

    if st.button("Search"):

        query_embedding = extract_feature(query_image)

        distances, indices = index.search(query_embedding, k)

        st.subheader("Retrieved Images")

        cols = st.columns(k)

        for i, idx in enumerate(indices[0]):

            # Original path stored in image_paths.npy
            image_path = str(image_paths[idx])

            # Convert Kaggle path to Hugging Face dataset path
            if "Stanford_Online_Products/" in image_path:
                relative_path = "Stanford_Online_Products/" + image_path.split(
                    "Stanford_Online_Products/"
                )[-1]
            else:
                relative_path = image_path

            try:
                # Download image from Hugging Face Dataset
                local_image = hf_hub_download(
                    repo_id=IMAGE_REPO_ID,
                    repo_type="dataset",
                    filename=relative_path,
                )

                result_image = Image.open(local_image).convert("RGB")

                with cols[i]:
                    st.image(
                        result_image,
                        caption=f"Similarity: {distances[0][i]:.4f}",
                        use_container_width=True,
                    )

            except Exception as e:
                with cols[i]:
                    st.error(f"Cannot load image\n\n{relative_path}\n\n{e}")
