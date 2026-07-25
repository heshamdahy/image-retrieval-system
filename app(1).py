
import io
import faiss
import numpy as np
import streamlit as st
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from datasets import load_dataset

ASSETS_REPO="heshamdahy/stanford-products-retrieval-assets"
DATASET_REPO="heshamdahy/stanford-products-images"

st.set_page_config(page_title="Image Retrieval",layout="wide")

@st.cache_resource
def load_assets():
    model_path=hf_hub_download(repo_id=ASSETS_REPO,filename="model.pth")
    index_path=hf_hub_download(repo_id=ASSETS_REPO,filename="index.faiss")
    emb_path=hf_hub_download(repo_id=ASSETS_REPO,filename="embeddings.npy")
    paths_path=hf_hub_download(repo_id=ASSETS_REPO,filename="image_paths_relative.npy")
    model=torch.load(model_path,map_location="cpu")
    model.eval()
    index=faiss.read_index(index_path)
    image_paths=np.load(paths_path,allow_pickle=True)
    embeddings=np.load(emb_path)
    return model,index,embeddings,image_paths

@st.cache_resource
def load_hf_dataset():
    return load_dataset(DATASET_REPO,split="train")

@st.cache_data
def get_image(rel_path):
    ds=load_hf_dataset()
    for x in ds:
        if x["path"]==rel_path:
            return x["image"]
    return None

def preprocess(img):
    from torchvision import transforms
    t=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t(img).unsqueeze(0)

def extract(model,img):
    with torch.no_grad():
        out=model(preprocess(img))
        if isinstance(out,(tuple,list)):
            out=out[0]
        return out.cpu().numpy().astype("float32")

model,index,embeddings,image_paths=load_assets()

st.title("Stanford Online Products Image Retrieval")
uploaded=st.file_uploader("Upload image",type=["jpg","jpeg","png"])
k=st.slider("Top K",1,20,5)

if uploaded:
    q=Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(q,width=250,caption="Query")
    feat=extract(model,q)
    D,I=index.search(feat,k)
    cols=st.columns(k)
    for c,idx,dist in zip(cols,I[0],D[0]):
        p=image_paths[idx]
        im=get_image(str(p))
        if im:
            c.image(im,caption=f"{dist:.2f}",use_container_width=True)
        else:
            c.write(p)
