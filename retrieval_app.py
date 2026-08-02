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
# لازم تكون موجودة لأن الموديل محفوظ كامل (pickled)
# ==========================

class open_clip_model(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.clip = model

        self.ffn = nn.Sequential(
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.clip.encode_image(x)
        x = self.ffn(x)
        x = F.normalize(x, dim=1)
        return x


# ==========================
# Load Assets
# ==========================

@st.cache_resource
def load_assets():

    # Model (open_clip.pth = موديل كامل محفوظ بالـ pickle)

    model_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="open_clip.pth"
    )

    # FAISS Flat Index (بحث دقيق، من غير HNSW)

    index_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="index_flat.faiss"
    )

    # Paths

    paths_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="image_paths_relative.npy"
    )

    # Load model
    # ملحوظة: لازم weights_only=False هنا لأن الملف فيه الموديل كامل
    # (مش state_dict بس)، وده هو سبب علامة "Suspicious" على Hugging Face.
    # طالما انت اللي عملت الملف ده بنفسك وتثق في مصدره، الأمر عادي.

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.to(device)
    model.eval()

    # Load FAISS

    index = faiss.read_index(index_path)

    # Load image paths

    image_paths = np.load(paths_path, allow_pickle=True)

    return model, index, image_paths


model, index, image_paths = load_assets()


# ==========================
# Image Transform
# (قيم CLIP الأصلية عشان تطابق الـ preprocessing وقت بناء الفهرس)
# ==========================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
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

    return embedding.astype("float32")


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

    image = Image.open(image_file).convert("RGB")

    return image


# ==========================
# UI
# ==========================

st.title("🛍️ Stanford Products Image Retrieval")

st.write(
    "Upload an image and retrieve similar products using OpenCLIP + FAISS"
)

uploaded_file = st.file_uploader(
    "Upload product image",
    type=["jpg", "jpeg", "png"]
)

k = st.slider(
    "Number of retrieved images",
    min_value=1,
    max_value=20,
    value=5
)

if uploaded_file:

    query_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Query Image")

    st.image(query_image, width=300)

    # تحقق سريع إن حجم الفهرس متطابق مع عدد مسارات الصور
    if index.ntotal != len(image_paths):
        st.warning(
            f"Mismatch: index has {index.ntotal} vectors, "
            f"but image_paths has {len(image_paths)} entries. "
            "Retrieved indices may be invalid."
        )

    if st.button("Search"):

        query_embedding = extract_feature(query_image)

        distances, indices = index.search(query_embedding, k)

        st.subheader("Retrieved Images")

        cols = st.columns(k)

        for i, idx in enumerate(indices[0]):

            # حماية من idx خارج حدود مصفوفة image_paths (أو -1)
            if idx < 0 or idx >= len(image_paths):
                with cols[i]:
                    st.error(
                        f"Index {idx} out of range "
                        f"(image_paths has {len(image_paths)} entries)"
                    )
                continue

            image_path = str(image_paths[idx])

            try:
                result_image = load_image_from_hf(image_path)

                with cols[i]:
                    st.image(
                        result_image,
                        caption=f"Distance: {distances[0][i]:.4f}",
                        use_container_width=True
                    )

            except Exception as e:
                with cols[i]:
                    st.error(f"Failed loading image\n{image_path}\n{e}")






    
   



   
   


   
               
