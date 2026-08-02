"""
Stanford Products Image Retrieval — Streamlit App
موديل: OpenCLIP + FAISS HNSW

ملاحظات مهمة عن التعديلات:
1) بدل ما نحمّل الموديل كامل بـ torch.load(weights_only=False) (اللي بيخلي
   ملف open_clip.pth يظهر "Suspicious" على Hugging Face)، دلوقتي بنحمّل
   state_dict بس، وبنبني الموديل من الكلاس يدويًا. ده أأمن وبيخلي الملف
   يتصنف Safe.
2) الـ Normalize بقى بقيم CLIP الأصلية (0.4815..., 0.4578..., 0.4082...)
   بدل قيم ImageNet، عشان لازم تتطابق مع الـ preprocessing اللي اتستخدم
   وقت بناء embeddings.npy / index_hnsw.faiss.
   *** لو انت أصلاً بنيت الـ embeddings بقيم ImageNet، رجّعها لـ ImageNet ***
   المهم إن الـ query preprocessing يطابق preprocessing الفهرسة تمامًا.
3) لازم يكون عندك ملف جديد اسمه open_clip_state_dict.pth مرفوع فيه
   state_dict بس (مش الموديل كامل). في الأسفل سكريبت منفصل لعمل التحويل
   من الملف القديم لو لسه معاك الملف الأصلي.
"""

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from PIL import Image

import numpy as np
import faiss

import open_clip

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

# اسم الـ backbone اللي اتدرب بيه open_clip الأساسي — عدّل الاسم والـ
# pretrained tag لو مختلفين عندك (مثلاً ViT-B-32 / openai)
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


# ==========================
# OpenCLIP Model Class
# لازم تكون موجودة لأن الموديل هيتبني بيها يدويًا
# ==========================

class OpenClipRetrievalModel(nn.Module):

    def __init__(self, clip_model, embed_dim: int = 512, out_dim: int = 256):
        super().__init__()

        self.clip = clip_model

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, out_dim)
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

    # --- تحميل الملفات من Hugging Face ---

    state_dict_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="open_clip_state_dict.pth"
    )

    index_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="index_hnsw.faiss"
    )

    paths_path = hf_hub_download(
        repo_id=ASSET_REPO,
        filename="image_paths_relative.npy"
    )

    # --- بناء الموديل من الصفر ثم تحميل الأوزان بس ---

    base_clip, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED
    )

    model = OpenClipRetrievalModel(base_clip)

    state_dict = torch.load(
        state_dict_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # --- تحميل الفهرس ---

    index = faiss.read_index(index_path)

    # --- تحميل مسارات الصور ---

    image_paths = np.load(paths_path, allow_pickle=True)

    return model, index, image_paths


model, index, image_paths = load_assets()


# ==========================
# Image Transform (CLIP normalization)
# ==========================

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ==========================
# Feature Extraction
# ==========================

def extract_feature(image: Image.Image) -> np.ndarray:

    image_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_t)

    embedding = embedding.cpu().numpy()

    return embedding.astype("float32")


# ==========================
# Download Image Cache
# ==========================

@st.cache_data
def load_image_from_hf(path: str) -> Image.Image:

    image_file = hf_hub_download(
        repo_id=IMAGE_REPO,
        repo_type="dataset",
        filename=path
    )

    return Image.open(image_file).convert("RGB")


# ==========================
# UI
# ==========================

st.title("🛍️ Stanford Products Image Retrieval")

st.write(
    "Upload an image and retrieve similar products using OpenCLIP + FAISS HNSW"
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

    if st.button("Search"):

        query_embedding = extract_feature(query_image)

        distances, indices = index.search(query_embedding, k)

        st.subheader("Retrieved Images")

        cols = st.columns(k)

        for i, idx in enumerate(indices[0]):

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
