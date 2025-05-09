import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "Unet"))
import streamlit as st
from unet import UNet
from streamlit_cropper import st_cropper
from PIL import Image
import torch
from dataloader import PreprocessData
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
@st.cache_resource
def preprocess_data(image):
    transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # standard normalization values for ImageNet
        ])
    image = transform(image)
    image_tensor = image.unsqueeze(0)
    return image_tensor

def predict(model, image_tensor, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No gradient calculation during testing
        
        image_tensor= image_tensor.to(device)
        
        # Forward pass
        outputs = model(image_tensor)
        # Assuming binary segmentation, apply sigmoid to the output
        predicted = torch.sigmoid(outputs) > 0.5  # Threshold for binary segmentation

    return predicted
def load_model(device):
    model = UNet(3, 6)
    model = model.to(device)
    model.load_state_dict(torch.load("pages/Unet/model.pth"))
    return model

def plot_images(image, predicted):
    fig, ax = plt.subplots(1, 7, figsize=(12, 8))
    mask_types = [
                'hair', 'skin', 
                'nose', 
                'l_eye','r_eye'
                ,'mouth']

    ax[0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
    ax[0].axis('off')
    ax[0].set_title("Image")
    for j in range(1,7):
        ax[j].imshow(predicted[0,j-1].cpu().numpy(), cmap='gray')
        ax[j].axis('off')
        ax[j].set_title(f"{mask_types[j-1]}")
    return fig

st.title("U-NetSegmentation")
st.write("This is a U-Net model that segments an image into different parts")
st.write("This model is trained on 32x32 images for only one epoch on the celebA dataset, with 0.96 accuracy with IoU metric.")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    ### Open the uploaded image
    image = Image.open(uploaded_file)
    st.write("## Crop Image")
    image = st_cropper(image, aspect_ratio=(1, 1))  # You can customize ratio
    st.image(image, caption="Cropped Image")
    ### Preprocess
    image = preprocess_data(image)
    st.write("## Prediction")
    ### Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    predicted = predict(model, image, device)
    ### Plot
    fig = plot_images(image, predicted)
    st.pyplot(fig)

    