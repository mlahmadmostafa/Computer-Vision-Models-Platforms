import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "GAN"))
import streamlit as st
from gan_model import Generator
import torch
import cv2
import torchvision.utils as vutils
import numpy as np


st.title("GAN Face generator")
st.write("This is a GAN model that generates faces from a random seed")
st.write("This model has 256 context weight 10 epochs on the celebA dataset")
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean
@st.cache_resource
def load_generator(device):
    G = Generator(z_dim=Z_DIM, img_channels=3, feature_g=256).to(device)
    G = G.to(device)
    G.load_state_dict(torch.load("pages/GAN/model.pth"))
    return G
@st.cache_data
def generate_image(seed: int, device="cuda"):
    torch.manual_seed(seed)
    z = torch.randn(16, Z_DIM, 1, 1).to(device)
    with torch.no_grad():
        fake_image = G(z)
        grid = vutils.make_grid(fake_image.detach(), nrow=4)  # No need for normalize=False
        grid = denormalize(grid, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Denormalize the [-1, 1] range
        img = grid.clamp(0, 1).mul(255).permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
        img = cv2.resize(img, (512, 512))
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number = st.slider("choose a seed", 0, 100)
st.write(f"Seed: {number}")

Z_DIM = 256

G = load_generator(device)

img = generate_image(number)
st.image(img)